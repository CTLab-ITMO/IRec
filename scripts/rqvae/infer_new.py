from loguru import logger
import os

import torch

import irec.callbacks as cb
from irec.data.dataloader import DataLoader
from irec.data.transforms import Collate, ToTorch, ToDevice
from irec.runners import InferenceRunner

from irec.utils import fix_random_seed

from data import EmbeddingDataset, process_embeddings
from models import NewRQVAE

SEED_VALUE = 42
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

BATCH_SIZE = 1024

INPUT_DIM = 4096
HIDDEN_DIM = 32
CODEBOOK_SIZE = 256
NUM_CODEBOOKS = 3

BETA = 0.25
MODEL_PATH = 'rqvae_beauty_new_best_0.0131.pth'
EXPERIMENT_NAME = 'rqvae_beauty_new'
IREC_PATH = '../../'


def main():
    fix_random_seed(SEED_VALUE)

    dataset = EmbeddingDataset(
        data_path=os.path.join(IREC_PATH, 'data/Beauty/content_embeddings.pkl')
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    ).map(Collate()).map(ToTorch()).map(ToDevice(DEVICE)).map(process_embeddings)

    model = NewRQVAE(
        input_dim=INPUT_DIM,
        num_codebooks=NUM_CODEBOOKS,
        codebook_size=CODEBOOK_SIZE,
        embedding_dim=HIDDEN_DIM,
        layers=[2048, 1024, 512, 256, 128],
        dropout_prob=0.1,
        beta=BETA,
        quant_loss_weight=1.0,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.debug(f'Overall parameters: {total_params:,}')
    logger.debug(f'Trainable parameters: {trainable_params:,}')

    callbacks = [
        cb.LoadModel(os.path.join(IREC_PATH, 'checkpoints', MODEL_PATH)),

        cb.BatchMetrics(metrics=lambda model_outputs, _: {
            'loss': model_outputs['loss'],
            'recon_loss': model_outputs['recon_loss'],
            'rqvae_loss': model_outputs['rqvae_loss'],
        }, name='valid'),

        cb.MetricAccumulator(
            accumulators={
                'valid/loss': cb.MeanAccumulator(),
                'valid/recon_loss': cb.MeanAccumulator(),
                'valid/rqvae_loss': cb.MeanAccumulator(),
            },
        ),

        cb.Logger().every_num_steps(len(dataloader)),

        cb.InferenceSaver(
            metrics=lambda batch, model_outputs, _: {'item_id': batch['item_id'], 'clusters': model_outputs['clusters']}, 
            save_path=os.path.join(IREC_PATH, 'results', f'{EXPERIMENT_NAME}_clusters.json'),
            format='json'
        )
    ]

    logger.debug('Everything is ready for training process!')

    runner = InferenceRunner(
        model=model,
        dataset=dataloader,
        callbacks=callbacks,
    )
    runner.run()


if __name__ == '__main__':
    main()
