import torch

import irec.callbacks as cb
from irec.runners import TrainingRunner, TrainingRunnerContext

class InitCodebooks(cb.TrainingCallback):
    def __init__(self, dataloader):
        super().__init__()
        self._dataloader = dataloader

    @torch.no_grad()
    def before_run(self, runner: TrainingRunner):
        for i in range(len(runner.model.codebooks)):
            X = next(iter(self._dataloader))['embedding']
            idx = torch.randperm(X.shape[0], device=X.device)[:len(runner.model.codebooks[i])]
            remainder = runner.model.encoder(X[idx])

            for j in range(i):
                codebook_indices = runner.model.get_codebook_indices(remainder, runner.model.codebooks[j])
                codebook_vectors = runner.model.codebooks[j][codebook_indices]
                remainder = remainder - codebook_vectors
            
            runner.model.codebooks[i].data = remainder.detach()


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
        for codebook_idx, codebook in enumerate(runner.model.codebooks):
            centroid_counts = torch.zeros(codebook.shape[0], dtype=torch.long, device=codebook.device)
            random_batch = next(iter(self._dataloader))['embedding']

            for batch in self._dataloader:                
                remainder = runner.model.encoder(batch['embedding'])
                for l in range(codebook_idx):
                    ind = runner.model.get_codebook_indices(remainder, runner.model.codebooks[l])
                    remainder = remainder - runner.model.codebooks[l][ind]
                
                indices = runner.model.get_codebook_indices(remainder, codebook)
                centroid_counts.scatter_add_(0, indices, torch.ones_like(indices))

            dead_mask = (centroid_counts == 0)
            num_dead = int(dead_mask.sum().item())
            num_fixed.append(num_dead)
            if num_dead == 0:
                continue

            remainder = runner.model.encoder(random_batch)
            for l in range(codebook_idx):
                ind = runner.model.get_codebook_indices(remainder, runner.model.codebooks[l])
                remainder = remainder - runner.model.codebooks[l][ind]
            remainder = remainder[torch.randperm(remainder.shape[0], device=codebook.device)][:num_dead]
            codebook[dead_mask] = remainder.detach()

        return num_fixed
