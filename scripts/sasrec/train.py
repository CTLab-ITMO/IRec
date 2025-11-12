from loguru import logger
import os

import torch

import irec.callbacks as cb
from irec.data.transforms import Collate, ToDevice
from irec.data.dataloader import DataLoader
from irec.models import AutoCast
from irec.runners import TrainingRunner
from irec.utils import fix_random_seed

from data import ArrowBatchDataset
from models import SasRecModel

SEED_VALUE = 42
DEVICE = 'cuda'

EXPERIMENT_NAME = 'sasrec_beauty'
NUM_EPOCHS = 300
MAX_SEQ_LEN = 20
TRAIN_BATCH_SIZE = 256
VALID_BATCH_SiZE = 256
EMBEDDING_DIM = 64
NUM_HEADS = 2
NUM_LAYERS = 2
FEEDFORWARD_DIM = 256
DROPOUT = 0.3
LR = 1e-4

NUM_ITEMS = 12101

IREC_PATH = '../../'

torch.set_float32_matmul_precision('high')
torch._dynamo.config.capture_scalar_outputs = True


def main():
    fix_random_seed(SEED_VALUE)

    train_dataloader = DataLoader(
        ArrowBatchDataset(
            os.path.join(IREC_PATH, 'data/Beauty/sasrec_train/'), 
            device='cpu', 
            preload=None
        ),
        batch_size=1, 
        shuffle=True, 
        num_workers=16,
        prefetch_factor=16,
        pin_memory=True, 
        persistent_workers=True,
        collate_fn=Collate()
    ).map(ToDevice(DEVICE))

    valid_dataloder = ArrowBatchDataset(
        os.path.join(IREC_PATH, 'data/Beauty/sasrec_valid/'),
        device=DEVICE,
        preload=True
    )

    eval_dataloder = ArrowBatchDataset(
        os.path.join(IREC_PATH, 'data/Beauty/sasrec_eval/'),
        device=DEVICE,
        preload=True
    )

    model = SasRecModel(
        num_items=NUM_ITEMS,
        max_sequence_length=MAX_SEQ_LEN,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dim_feedforward=FEEDFORWARD_DIM,
        activation='relu',
        topk_k=20,
        dropout=DROPOUT,
        layer_norm_eps=1e-8,
        initializer_range=0.02
    )
    model = torch.compile(model, mode="default", fullgraph=False)
    model = model.to('cuda')

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.debug(f'Overall parameters: {total_params:,}')
    logger.debug(f'Trainable parameters: {trainable_params:,}')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
    )

    EPOCH_NUM_STEPS = int(len(train_dataloader) // NUM_EPOCHS)

    callbacks = [
        cb.BatchMetrics(metrics=lambda model_outputs, _: {
            'loss': model_outputs['loss'].item(),
            'recall@5': model_outputs['recall@5'].tolist(),
            'recall@10': model_outputs['recall@10'].tolist(),
            'recall@20': model_outputs['recall@20'].tolist(),
            'ndcg@5': model_outputs['ndcg@5'].tolist(),
            'ndcg@10': model_outputs['ndcg@10'].tolist(),
            'ndcg@20': model_outputs['ndcg@20'].tolist(),
        }, name='train'),
        cb.MetricAccumulator(
            accumulators={
                'train/loss': cb.MeanAccumulator(),
                'train/recall@5': cb.MeanAccumulator(),
                'train/recall@10': cb.MeanAccumulator(),
                'train/recall@20': cb.MeanAccumulator(),
                'train/ndcg@5': cb.MeanAccumulator(),
                'train/ndcg@10': cb.MeanAccumulator(),
                'train/ndcg@20': cb.MeanAccumulator(),
            },
            reset_every_num_steps=EPOCH_NUM_STEPS
        ),

        cb.Validation(
            dataset=valid_dataloder,
            callbacks=[
                cb.BatchMetrics(metrics=lambda model_outputs, _: {
                    'loss': model_outputs['loss'].item(),
                    'recall@5': model_outputs['recall@5'].tolist(),
                    'recall@10': model_outputs['recall@10'].tolist(),
                    'recall@20': model_outputs['recall@20'].tolist(),
                    'ndcg@5': model_outputs['ndcg@5'].tolist(),
                    'ndcg@10': model_outputs['ndcg@10'].tolist(),
                    'ndcg@20': model_outputs['ndcg@20'].tolist(),
                }, name='validation'),
                cb.MetricAccumulator(
                    accumulators={
                        'validation/loss': cb.MeanAccumulator(),
                        'validation/recall@5': cb.MeanAccumulator(),
                        'validation/recall@10': cb.MeanAccumulator(),
                        'validation/recall@20': cb.MeanAccumulator(),
                        'validation/ndcg@5': cb.MeanAccumulator(),
                        'validation/ndcg@10': cb.MeanAccumulator(),
                        'validation/ndcg@20': cb.MeanAccumulator(),
                    },
                ),                
            ],
        ).every_num_steps(EPOCH_NUM_STEPS),

        cb.Validation(
            dataset=eval_dataloder,
            callbacks=[
                cb.BatchMetrics(metrics=lambda model_outputs, _: {
                    'loss': model_outputs['loss'].item(),
                    'recall@5': model_outputs['recall@5'].tolist(),
                    'recall@10': model_outputs['recall@10'].tolist(),
                    'recall@20': model_outputs['recall@20'].tolist(),
                    'ndcg@5': model_outputs['ndcg@5'].tolist(),
                    'ndcg@10': model_outputs['ndcg@10'].tolist(),
                    'ndcg@20': model_outputs['ndcg@20'].tolist(),
                }, name='eval'),
                cb.MetricAccumulator(
                    accumulators={
                        'eval/loss': cb.MeanAccumulator(),
                        'eval/recall@5': cb.MeanAccumulator(),
                        'eval/recall@10': cb.MeanAccumulator(),
                        'eval/recall@20': cb.MeanAccumulator(),
                        'eval/ndcg@5': cb.MeanAccumulator(),
                        'eval/ndcg@10': cb.MeanAccumulator(),
                        'eval/ndcg@20': cb.MeanAccumulator(),
                    },
                ),
            ],
        ).every_num_steps(EPOCH_NUM_STEPS),
        
        cb.Logger().every_num_steps(EPOCH_NUM_STEPS),
        cb.TensorboardLogger(experiment_name=EXPERIMENT_NAME, logdir=os.path.join(IREC_PATH, 'tensorboard_logs')),

        cb.EarlyStopping(
            metric='validation/ndcg@20', 
            patience=40,
            minimize=False,
            model_path=os.path.join(IREC_PATH, 'checkpoints', EXPERIMENT_NAME)
        ).every_num_steps(EPOCH_NUM_STEPS)

        # cb.Profiler(
        #     wait=10,
        #     warmup=10,
        #     active=10,
        #     logdir=os.path.join(IREC_PATH, 'tensorboard_logs')
        # ),
        # cb.StopAfterNumSteps(40)
    ]

    logger.debug('Everything is ready for training process!')

    runner = TrainingRunner(
        model=AutoCast(model, dtype=torch.bfloat16, device_type=DEVICE),
        optimizer=optimizer,
        dataset=train_dataloader,
        callbacks=callbacks,
    )
    runner.run()


if __name__ == '__main__':
    main()
