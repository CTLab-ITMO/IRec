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

SEED_VALUE = 42
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

BATCH_SIZE = 1024

INPUT_DIM = 4096
HIDDEN_DIM = 32
CODEBOOK_SIZE = 256
NUM_CODEBOOKS = 3

BETA = 0.25
MODEL_PATH = 'rqvae_beauty_best_8_1_1_cf_best_0.0079.pth'

EXPERIMENT_NAME = 'rqvae_beauty_best_8_1_1_cf'
IREC_PATH = '../../'


def main():
    fix_random_seed(SEED_VALUE)

    dataset = EmbeddingDataset(
        data_path=os.path.join(IREC_PATH, 'data/Beauty/content_embeddings_train_only_0.8_0.1_0.1.pkl')
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    ).map(Collate()).map(ToTorch()).map(ToDevice(DEVICE)).map(ProcessEmbeddings(embedding_dim=INPUT_DIM, keys=['embedding']))

    cf_embeddings = torch.load(
        os.path.join(IREC_PATH, 'results', 'sasrec_beauty_8_1_1_32d_item_embeddings.pt'), 
        map_location='cpu'
    )['weight']

    model = BestRQVAE(
        input_dim=INPUT_DIM,
        num_codebooks=NUM_CODEBOOKS,
        codebook_size=CODEBOOK_SIZE,
        embedding_dim=HIDDEN_DIM,
        beta=BETA,
        quant_loss_weight=1.0,
        cf_loss_weight=1.0,
        cf_embeddings=cf_embeddings
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
    
    with open(os.path.join(IREC_PATH, 'results', f'{EXPERIMENT_NAME}_clusters_colisionless.json'), 'w') as f:
        json.dump(inter, f, indent=2)


if __name__ == '__main__':
    main()
