from loguru import logger
import os

import torch

import irec.callbacks as cb
from irec.data.dataloader import DataLoader
from irec.data.transforms import Collate, ToTorch, ToDevice
from irec.runners import InferenceRunner

from irec.utils import fix_random_seed

from data import EmbeddingDataset, ProcessEmbeddings
from models import BestRQVAE

from train_original import (
    SEED_VALUE, DEVICE, BATCH_SIZE, INPUT_DIM, HIDDEN_DIM,
    CODEBOOK_SIZE, NUM_CODEBOOKS, BETA, EXPERIMENT_NAME, IREC_PATH,
)
from pathlib import Path


def main():
    fix_random_seed(SEED_VALUE)
    os.makedirs(os.path.join(IREC_PATH, 'results'), exist_ok=True)
    candidates = list((Path(IREC_PATH) / 'checkpoints').glob(EXPERIMENT_NAME + '_best_*.pth'))
    assert len(candidates) == 1, f'Expected one best checkpoint, found {candidates}'
    model_path = candidates[0]

    dataset = EmbeddingDataset(
        data_path=os.path.join(IREC_PATH, 'data/Beauty/content_embeddings_original.pkl')
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    ).map(Collate()).map(ToTorch()).map(ToDevice(DEVICE)).map(ProcessEmbeddings(embedding_dim=INPUT_DIM, keys=['embedding']))

    model = BestRQVAE(
        input_dim=INPUT_DIM,
        num_codebooks=NUM_CODEBOOKS,
        codebook_size=CODEBOOK_SIZE,
        embedding_dim=HIDDEN_DIM,
        beta=BETA,
        quant_loss_weight=1.0,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.debug(f'Overall parameters: {total_params:,}')
    logger.debug(f'Trainable parameters: {trainable_params:,}')

    callbacks = [
        cb.LoadModel(str(model_path)),

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

    import json
    from collections import defaultdict
    import numpy as np

    with open(os.path.join(IREC_PATH, 'results', f'{EXPERIMENT_NAME}_clusters.json'), 'r') as f:
        mappings = json.load(f)
    
    inter = {}
    sem_2_ids = defaultdict(list)
    for mapping in mappings:
        item_id = mapping['item_id']
        clusters = mapping['clusters']
        inter[int(item_id)] = clusters
        sem_2_ids[tuple(clusters)].append(int(item_id))

    for semantics, items in sem_2_ids.items():
        assert len(items) <= CODEBOOK_SIZE, str(len(items))
        collision_solvers = np.random.permutation(CODEBOOK_SIZE)[:len(items)].tolist()
        for item_id, collision_solver in zip(items, collision_solvers):
            inter[item_id].append(collision_solver)
            for i in range(len(inter[item_id])):
                inter[item_id][i] += CODEBOOK_SIZE * i
    
    assert set(inter) == set(range(len(dataset)))
    assert len({tuple(sid) for sid in inter.values()}) == len(inter)
    with open(os.path.join(IREC_PATH, 'results', 'rqvae_beauty_original_semantic_ids.json'), 'w') as f:
        json.dump(inter, f, indent=2)


if __name__ == '__main__':
    main()
