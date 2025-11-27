import torch
import torch.nn as nn
import torch.nn.functional as F

class PlumRQVAE(nn.Module):
    def __init__(
            self,
            input_dim,
            num_codebooks,
            codebook_size,
            embedding_dim,
            beta=0.25,
            quant_loss_weight=1.0,
            contrastive_loss_weight=1.0,
            temperature=0.0,
    ):
        super().__init__()
        self.register_buffer('beta', torch.tensor(beta))
        self.temperature = temperature

        self.input_dim = input_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.quant_loss_weight = quant_loss_weight

        self.contrastive_loss_weight = contrastive_loss_weight

        self.encoder = self.make_encoding_tower(input_dim, embedding_dim)
        self.decoder = self.make_encoding_tower(embedding_dim, input_dim)

        self.codebooks = torch.nn.ParameterList()
        for _ in range(num_codebooks):
            cb = torch.FloatTensor(codebook_size, embedding_dim)
            #nn.init.normal_(cb)
            self.codebooks.append(cb)

    @staticmethod
    def make_encoding_tower(d1, d2, bias=False):
        return torch.nn.Sequential(
            nn.Linear(d1, d1),
            nn.ReLU(),
            nn.Linear(d1, d2),
            nn.ReLU(),
            nn.Linear(d2, d2, bias=bias)
        )

    @staticmethod
    def get_codebook_indices(remainder, codebook):
        dist = torch.cdist(remainder, codebook)
        return dist.argmin(dim=-1)

    def _quantize_representation(self, latent_vector):
        latent_restored = 0
        remainder = latent_vector

        for codebook in self.codebooks:
            codebook_indices = self.get_codebook_indices(remainder, codebook)
            quantized = codebook[codebook_indices]
            codebook_vectors = remainder + (quantized - remainder).detach()
            latent_restored += codebook_vectors
            remainder = remainder - codebook_vectors

        return latent_restored

    def contrastive_loss(self, p_i, p_i_star):
        N_b = p_i.size(0)

        p_i = F.normalize(p_i, p=2, dim=-1) #TODO посмотреть без нормалайза
        p_i_star = F.normalize(p_i_star, p=2, dim=-1)

        similarities = torch.matmul(p_i, p_i_star.T) / self.temperature

        labels = torch.arange(N_b, dtype=torch.long, device=p_i.device)

        loss = F.cross_entropy(similarities, labels)

        return loss

    def forward(self, inputs):
        latent_vector = self.encoder(inputs['embedding'])
        item_ids = inputs['item_id']

        latent_restored = 0
        rqvae_loss = 0
        clusters = []
        remainder = latent_vector

        for codebook in self.codebooks:
            codebook_indices = self.get_codebook_indices(remainder, codebook)
            clusters.append(codebook_indices)

            quantized = codebook[codebook_indices]
            codebook_vectors = remainder + (quantized - remainder).detach()

            rqvae_loss += self.beta * torch.nn.functional.mse_loss(remainder, quantized.detach())
            rqvae_loss += torch.nn.functional.mse_loss(quantized, remainder.detach())

            latent_restored += codebook_vectors
            remainder = remainder - codebook_vectors

        embeddings_restored = self.decoder(latent_restored)
        recon_loss = torch.nn.functional.mse_loss(embeddings_restored, inputs['embedding'])

        if 'cooccurrence_embedding' in inputs:
            cooccurrence_latent = self.encoder(inputs['cooccurrence_embedding'].to(latent_restored.device))
            cooccurrence_restored = self._quantize_representation(cooccurrence_latent)
            con_loss = self.contrastive_loss(latent_restored, cooccurrence_restored)
        else:
            con_loss = torch.as_tensor(0.0, device=latent_vector.device)

        loss = (
                recon_loss
                + self.quant_loss_weight * rqvae_loss
                + self.contrastive_loss_weight * con_loss
        ).mean()

        clusters_counts = []
        for cluster in clusters:
            clusters_counts.append(torch.bincount(cluster, minlength=self.codebook_size))

        return loss, {
            'loss': loss.item(),
            'recon_loss': recon_loss.mean().item(),
            'rqvae_loss': rqvae_loss.mean().item(),
            'con_loss': con_loss.item(),

            'clusters_counts': clusters_counts,
            'clusters': torch.stack(clusters).T,
            'embedding_hat': embeddings_restored,
        }