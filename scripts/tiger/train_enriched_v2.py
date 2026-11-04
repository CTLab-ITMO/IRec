from functools import partial
import json
from loguru import logger
import os

import torch

import irec.callbacks as cb
from irec.data.transforms import Collate, ToDevice, ToTorch
from irec.data.dataloader import DataLoader
from irec.runners import TrainingRunner
from irec.utils import fix_random_seed

from data import Dataset
from processing import UserHashing, SemanticIdsMapper, ToMasked, TigerProcessing
from models import TigerModel, CorrectItemsLogitsProcessor

SEED_VALUE = 42
DEVICE = 'cuda'

NUM_EPOCHS = 300
MAX_SEQ_LEN = 20
TRAIN_BATCH_SIZE = 256
VALID_BATCH_SIZE = 16
EMBEDDING_DIM = 128
CODEBOOK_SIZE = 256
NUM_POSITIONS = 20
NUM_USER_HASH = 2000
NUM_HEADS = 6
NUM_LAYERS = 4
FEEDFORWARD_DIM = 1024
KV_DIM = 64
DROPOUT = 0.1
NUM_BEAMS = 30
TOP_K = 20
NUM_CODEBOOKS = 4
LR = 3e-4

EXPERIMENT_NAME = (f'tiger_beauty_enriched_v2_seed{SEED_VALUE}_bs{TRAIN_BATCH_SIZE}'
                   f'_dim{EMBEDDING_DIM}_heads{NUM_HEADS}_layers{NUM_LAYERS}'
                   f'_ff{FEEDFORWARD_DIM}_drop{DROPOUT}_lr{LR}')
IREC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

torch.set_float32_matmul_precision('high')
torch._dynamo.config.capture_scalar_outputs = True



def main():
    fix_random_seed(SEED_VALUE)
    os.makedirs(os.path.join(IREC_PATH, "checkpoints"), exist_ok=True)

    with open(os.path.join(IREC_PATH, 'results/rqvae_beauty_enriched_v2_semantic_ids.json'), 'r') as f:
        mappings = json.load(f)
    
    data = Dataset.create(
        inter_json_path=os.path.join(IREC_PATH, 'data/Beauty/inter.json'),
        max_sequence_length=MAX_SEQ_LEN, sampler_type='tiger', is_extended=True,
    )
    train_data, valid_data, eval_data = data.get_datasets()
    assert set(mappings) == {str(i) for i in range(data.num_items)}

    def make_loader(dataset, batch_size, training=False):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=training,
                            drop_last=training).map(Collate())
        loader = loader.map(UserHashing(NUM_USER_HASH))
        loader = loader.map(SemanticIdsMapper(mappings, names=['item', 'labels']))
        loader = loader.map(ToMasked('item.semantic', is_right_aligned=True))
        loader = loader.map(ToMasked('labels.semantic', is_right_aligned=True))
        if not training:
            loader = loader.map(ToMasked('visited', is_right_aligned=True))
        return loader.map(TigerProcessing()).map(ToTorch()).map(ToDevice(DEVICE))

    train_dataloader = make_loader(train_data, TRAIN_BATCH_SIZE, training=True).repeat(NUM_EPOCHS)
    valid_dataloder = make_loader(valid_data, VALID_BATCH_SIZE)
    eval_dataloder = make_loader(eval_data, VALID_BATCH_SIZE)

    model = TigerModel(
        embedding_dim=EMBEDDING_DIM,
        codebook_size=CODEBOOK_SIZE,
        sem_id_len=NUM_CODEBOOKS,
        user_ids_count=NUM_USER_HASH,
        num_positions=NUM_POSITIONS,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_LAYERS,
        num_decoder_layers=NUM_LAYERS,
        dim_feedforward=FEEDFORWARD_DIM,
        num_beams=NUM_BEAMS,
        num_return_sequences=TOP_K,
        activation='relu',
        d_kv=KV_DIM,
        dropout=DROPOUT,
        layer_norm_eps=1e-6,
        initializer_range=0.02,
        logits_processor=partial(
            CorrectItemsLogitsProcessor,
            NUM_CODEBOOKS,
            CODEBOOK_SIZE,
            mappings,
            NUM_BEAMS
        )
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.debug(f'Overall parameters: {total_params:,}')
    logger.debug(f'Trainable parameters: {trainable_params:,}')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
    )

    EPOCH_NUM_STEPS = len(train_dataloader) // NUM_EPOCHS
    assert EPOCH_NUM_STEPS > 0

    callbacks = [
        cb.BatchMetrics(metrics=lambda model_outputs, _: {
            'loss': model_outputs['loss'].item(),
        }, name='train'),
        cb.MetricAccumulator(
            accumulators={
                'train/loss': cb.MeanAccumulator(),
            },
            reset_every_num_steps=EPOCH_NUM_STEPS
        ),

        cb.Validation(
            dataset=valid_dataloder,
            callbacks=[
                cb.BatchMetrics(metrics=lambda model_outputs, _:{
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
        model=model,
        optimizer=optimizer,
        dataset=train_dataloader,
        callbacks=callbacks,
    )
    runner.run()


if __name__ == '__main__':
    main()
