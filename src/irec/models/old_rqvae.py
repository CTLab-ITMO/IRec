import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.norm = nn.LayerNorm(input_dim)
        self.layer = nn.Linear(input_dim, output_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedding = x
        embedding = self.norm(embedding)
        embedding = self.layer(embedding)
        embedding = self.act(embedding)
        embedding = self.dropout(embedding)

        if self.input_dim == self.output_dim:
            return embedding + x
        return embedding


class Tower(nn.Module):
    def __init__(self, dims, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(ResidualBlock(dims[i], dims[i + 1], dropout))
    
    def forward(self, x):
        embedding = x
        for layer in self.layers:
            embedding = layer(embedding)
        return embedding


class RQVAE(nn.Module):
    def __init__(
            self,
            input_dim,
            num_codebooks,
            codebook_size,
            embedding_dim,
            layers,
            dropout_prob=0.0,
            beta=0.25,
            quant_loss_weight=1.0,
            
        ):
        super().__init__()

        self.input_dim = input_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.quant_loss_weight = quant_loss_weight

        self.layers = layers
        self.dropout_prob = dropout_prob

        self.encoder_layer_dims = [self.input_dim] + self.layers + [self.embedding_dim]
        self.decoder_layer_dims = self.encoder_layer_dims[::-1]

        # TODO add inizialisation with AE
        self.encoder = Tower(
            dims=self.encoder_layer_dims,
            dropout=self.dropout_prob
        )
        self.decoder = Tower(
            dims=self.decoder_layer_dims,
            dropout=self.dropout_prob
        )

        self.codebooks = torch.nn.ParameterList()
        for _ in range(num_codebooks):
            cb = torch.FloatTensor(codebook_size, embedding_dim)
            self.codebooks.append(cb)

    @staticmethod
    def make_encoding_tower(d1, d2, bias=False):
        return torch.nn.Sequential(
            nn.LayerNorm(d1),
            nn.Linear(d1, d1),
            nn.GELU(),
            torch.nn.Linear(d1, d2, bias=bias)
        )

    @staticmethod
    def get_codebook_indices(remainder, codebook):
        dist = torch.cdist(remainder, codebook)
        return dist.argmin(dim=-1)

    def forward(self, inputs):
        latent_vector = self.encoder(inputs['embedding'])

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
        recon_loss = F.mse_loss(embeddings_restored, inputs['embedding'])
        loss = (recon_loss + self.quant_loss_weight * rqvae_loss).mean()

        clusters_counts = []
        for cluster in clusters:
            clusters_counts.append(torch.bincount(cluster, minlength=self.codebook_size))

        return loss, {
            'loss': loss.item(),
            'recon_loss': recon_loss.mean().item(),
            'rqvae_loss': rqvae_loss.mean().item(),

            'clusters_counts': clusters_counts,
            'clusters': torch.stack(clusters).T,
            'embedding_hat': embeddings_restored,
        }
