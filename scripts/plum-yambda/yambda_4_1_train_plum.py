from loguru import logger
import os

import torch

import pickle

import irec.callbacks as cb
from irec.data.dataloader import DataLoader
from irec.data.transforms import Collate, ToTorch, ToDevice
from irec.runners import TrainingRunner

from irec.utils import fix_random_seed

from callbacks import InitCodebooks, FixDeadCentroids
from data import EmbeddingDatasetParquet, ProcessEmbeddings
from models import PlumRQVAE
from transforms import AddWeightedCooccurrenceEmbeddingsVectorized
from cooc_data import CoocMappingDataset

SEED_VALUE = 42
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

NUM_EPOCHS = 35
BATCH_SIZE = 1024

MAX_NEIGHBOURS_COUNT = 1000

INPUT_DIM = 128
HIDDEN_DIM = 32
CODEBOOK_SIZE = 256
NUM_CODEBOOKS = 3
BETA = 0.25
LR = 1e-4
WINDOW_SIZE = 2

EXPERIMENT_NAME = f'4-1_filtered_yambda_gpu_week_ws_{WINDOW_SIZE}'
INTER_TRAIN_PATH = "/home/jovyan/IRec/data/Yambda/week-splits/merged_for_exps_filtered/exp_4-1_0.9_inter_semantics_train.json" #отсекать старое (может и нет)
EMBEDDINGS_PATH = "/home/jovyan/IRec/sigir/yambda_data/yambda_embeddings_reindexed.parquet"
IREC_PATH = '../../'

print(INTER_TRAIN_PATH)
def main():
    fix_random_seed(SEED_VALUE)

    data = CoocMappingDataset.create_from_split_part(
        train_inter_json_path=INTER_TRAIN_PATH,
        window_size=WINDOW_SIZE
    )

    dataset = EmbeddingDatasetParquet(
        data_path=EMBEDDINGS_PATH
    )

    item_id_to_embedding = {}
    all_item_ids = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        item_id = int(sample['item_id'])
        item_id_to_embedding[item_id] = torch.tensor(sample['embedding'], device=DEVICE)
        all_item_ids.append(item_id)

    add_cooc_transform = AddWeightedCooccurrenceEmbeddingsVectorized(
            cooccur_counts=data.cooccur_counter_mapping,
            item_id_to_embedding=item_id_to_embedding,
            all_item_ids=all_item_ids,
            device=DEVICE,
            limit_neighbors=True,
            max_neighbors = MAX_NEIGHBOURS_COUNT
    )

    train_dataloader = DataLoader( #call в основном потоке делается нужно исправить
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    ).map(Collate()).map(ToTorch()).map(ToDevice(DEVICE)).map(
        ProcessEmbeddings(embedding_dim=INPUT_DIM, keys=['embedding'])
    ).map(add_cooc_transform
    ).repeat(NUM_EPOCHS)

    valid_dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    ).map(Collate()).map(ToTorch()).map(ToDevice(DEVICE)).map(ProcessEmbeddings(embedding_dim=INPUT_DIM, keys=['embedding'])
    ).map(add_cooc_transform)

    LOG_EVERY_NUM_STEPS = int(len(train_dataloader) // NUM_EPOCHS)

    model = PlumRQVAE(
        input_dim=INPUT_DIM,
        num_codebooks=NUM_CODEBOOKS,
        codebook_size=CODEBOOK_SIZE,
        embedding_dim=HIDDEN_DIM,
        beta=BETA,
        quant_loss_weight=1.0,
        contrastive_loss_weight=1.0,
        temperature=1.0
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
            'con_loss': model_outputs['con_loss']
        }, name='train'),

        FixDeadCentroids(valid_dataloader),

        cb.MetricAccumulator(
            accumulators={
                'train/loss': cb.MeanAccumulator(),
                'train/recon_loss': cb.MeanAccumulator(),
                'train/rqvae_loss': cb.MeanAccumulator(),
                'train/con_loss': cb.MeanAccumulator(),
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
                    'con_loss': model_outputs['con_loss']
                }, name='valid'),
                cb.MetricAccumulator(
                    accumulators={
                        'valid/loss': cb.MeanAccumulator(),
                        'valid/recon_loss': cb.MeanAccumulator(),
                        'valid/rqvae_loss': cb.MeanAccumulator(),
                        'valid/con_loss': cb.MeanAccumulator()
                    }
                ),
            ],
        ).every_num_steps(LOG_EVERY_NUM_STEPS),

        cb.Logger().every_num_steps(LOG_EVERY_NUM_STEPS),
        cb.TensorboardLogger(experiment_name=EXPERIMENT_NAME, logdir=os.path.join(IREC_PATH, 'tensorboard_logs')),

        cb.Profiler(
            wait=10,
            warmup=10,
            active=10,
            logdir=os.path.join(IREC_PATH, 'tensorboard_logs')
        ),

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
