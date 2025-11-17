from loguru import logger
import os

import torch

import irec.callbacks as cb
from irec.data.dataloader import DataLoader
from irec.data.transforms import Collate, ToTorch, ToDevice
from irec.runners import TrainingRunner, TrainingRunnerContext

from irec.utils import fix_random_seed

from callbacks import InitCodebooks, FixDeadCentroids
from data import EmbeddingDataset, process_embeddings
from models import OldRQVAE, NewRQVAE, BestRQVAE

SEED_VALUE = 42
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

NUM_EPOCHS = 200
BATCH_SIZE = 1024

INPUT_DIM = 4096
HIDDEN_DIM = 32
CODEBOOK_SIZE = 256
NUM_CODEBOOKS = 3
BETA = 0.25
LR = 1e-4

EXPERIMENT_NAME = 'rqvae_beauty_new'
IREC_PATH = '../../'



class InitCodebooks(cb.TrainingCallback):
    def __init__(self, dataloader):
        super().__init__()
        self._dataloader = dataloader

    @torch.no_grad()
    def before_run(self, runner: TrainingRunner):
        for i in range(len(runner.model.rq.vq_layers)):
            X = next(iter(self._dataloader))['embedding']
            idx = torch.randperm(X.shape[0], device=X.device)[:runner.model.codebook_size]
            remainder = runner.model.encoder(X[idx])

            for j in range(i):
                codebook_indices = runner.model.get_codebook_indices(remainder, runner.model.rq.vq_layers[j])
                codebook_vectors = runner.model.rq.vq_layers[j].embedding(codebook_indices)
                remainder = remainder - codebook_vectors
            
            runner.model.rq.vq_layers[i].embedding.weight.data = remainder.detach()


class FixDeadCentroids(cb.TrainingCallback):
    def __init__(self, dataloader):
        super().__init__()
        self._dataloader = dataloader

    def after_step(self, runner: TrainingRunner, context: TrainingRunnerContext):
        for i, num_fixed in enumerate(self.fix_dead_codebooks(runner)):
            context.metrics[f'num_dead/{i}'] = num_fixed

    @torch.no_grad()
    def fix_dead_codebooks(self, runner: TrainingRunner):
        num_fixed = []
        for codebook_idx, quantizer in enumerate(runner.model.rq.vq_layers):
            centroid_counts = torch.zeros(quantizer.codebook_size, dtype=torch.long, device=DEVICE)
            random_batch = next(iter(self._dataloader))['embedding']

            for batch in self._dataloader:            
                remainder = runner.model.encoder(batch['embedding'])
                for l in range(codebook_idx):
                    _, _, ind = runner.model.rq.vq_layers[l](remainder)
                    remainder = remainder - runner.model.rq.vq_layers[l].embedding(ind)

                indices = runner.model.get_codebook_indices(remainder, quantizer)
                centroid_counts.scatter_add_(0, indices, torch.ones_like(indices))

            dead_mask = (centroid_counts == 0)
            num_dead = int(dead_mask.sum().item())
            num_fixed.append(num_dead)
            if num_dead == 0:
                continue

            remainder = runner.model.encoder(random_batch)
            for l in range(codebook_idx):
                ind = runner.model.get_codebook_indices(remainder, runner.model.rq.vq_layers[l])
                remainder = remainder - runner.model.rq.vq_layers[l].embedding(ind)
            remainder = remainder[torch.randperm(remainder.shape[0], device=remainder.device)][:num_dead]
            runner.model.rq.vq_layers[codebook_idx].embedding.weight.data[dead_mask] = remainder.detach()

        return num_fixed



def main():
    fix_random_seed(SEED_VALUE)

    dataset = EmbeddingDataset(
        data_path=os.path.join(IREC_PATH, 'data/Beauty/content_embeddings.pkl')
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    ).map(Collate()).map(ToTorch()).map(ToDevice(DEVICE)).map(process_embeddings).repeat(NUM_EPOCHS)

    valid_dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    ).map(Collate()).map(ToTorch()).map(ToDevice(DEVICE)).map(process_embeddings)

    # index_dataloader = DataLoader(
    #     dataset,
    #     batch_size=len(dataset),
    #     shuffle=False,
    #     drop_last=False,
    # ).map(Collate()).map(ToTorch()).map(ToDevice(DEVICE)).map(process_embeddings)

    # cf_embedding_path = '../data/Beauty/collaborative_item_embeddings.pt'
    # if cf_embedding_path is not None:
    #     cf_embeddings = torch.load(cf_embedding_path).squeeze().detach().numpy()

    LOG_EVERY_NUM_STEPS = int(len(train_dataloader) // NUM_EPOCHS)

    model = NewRQVAE(
        input_dim=INPUT_DIM,
        num_codebooks=NUM_CODEBOOKS,
        codebook_size=CODEBOOK_SIZE,
        embedding_dim=HIDDEN_DIM,
        layers=[2048, 1024, 512, 256, 128],
        dropout_prob=0.1,
        beta=BETA,
        quant_loss_weight=1.0,
        # cf_loss_weight=0.0,
        # cf_embeddings=cf_embeddings
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.debug(f'Overall parameters: {total_params:,}')
    logger.debug(f'Trainable parameters: {trainable_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, fused=True)

    callbacks = [
        InitCodebooks(valid_dataloader),

        cb.BatchMetrics(metrics=lambda model_outputs, batch: {
            'loss': model_outputs['loss'],
            'recon_loss': model_outputs['recon_loss'],
            'rqvae_loss': model_outputs['rqvae_loss'],
        }, name='train'),
        
        FixDeadCentroids(valid_dataloader),

        cb.MetricAccumulator(
            accumulators={
                'train/loss': cb.MeanAccumulator(),
                'train/recon_loss': cb.MeanAccumulator(),
                'train/rqvae_loss': cb.MeanAccumulator(),
                # 'train/cf_loss': MeanAccumulator(),
                'num_dead/0': cb.MeanAccumulator(),
                'num_dead/1': cb.MeanAccumulator(),
                'num_dead/2': cb.MeanAccumulator(),
            },
            reset_every_num_steps=LOG_EVERY_NUM_STEPS
        ),

        cb.Validation(
            dataset=valid_dataloader,
            callbacks=[
                cb.BatchMetrics(metrics=lambda model_outputs, batch: {
                    'loss': model_outputs['loss'],
                    'recon_loss': model_outputs['recon_loss'],
                    'rqvae_loss': model_outputs['rqvae_loss'],
                }, name='valid'),
                cb.MetricAccumulator(
                    accumulators={
                        'valid/loss': cb.MeanAccumulator(),
                        'valid/recon_loss': cb.MeanAccumulator(),
                        'valid/rqvae_loss': cb.MeanAccumulator(),
                        # 'valid/cf_loss': MeanAccumulator(),
                    }
                ),
            ],
        ).every_num_steps(LOG_EVERY_NUM_STEPS),

        cb.Logger().every_num_steps(LOG_EVERY_NUM_STEPS),
        cb.TensorboardLogger(experiment_name=EXPERIMENT_NAME, logdir=os.path.join(IREC_PATH, 'tensorboard_logs')),

        cb.EarlyStopping(
            metric='valid/recon_loss',
            patience=40,
            minimize=True,
            model_path=os.path.join(IREC_PATH, 'checkpoints', EXPERIMENT_NAME)
        ).every_num_steps(LOG_EVERY_NUM_STEPS),
    ]

    logger.debug('Everything is ready for training process!')

    runner = TrainingRunner(
        model=model,
        optimizer=optimizer,
        dataset=train_dataloader,
        callbacks=callbacks,
    )
    runner.run()


if __name__ == '__main__':
    main()
