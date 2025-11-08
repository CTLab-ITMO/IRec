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


class VectorQuantizer(nn.Module):
    
    def __init__(
            self, 
            codebook_size,
            embedding_dim,
            mu=0.25,
        ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.mu = mu

        self.embedding = nn.Embedding(self.codebook_size, self.embedding_dim)

    def get_codebook(self):
        return self.embedding.weight
    
    def forward(self, latent_embeddings):
        # Get closest centroids
        d = torch.sum(latent_embeddings**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t() - 2 * torch.matmul(latent_embeddings, self.embedding.weight.t())
        indices = torch.argmin(d, dim=-1)

        x_q = self.embedding(indices)

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), latent_embeddings)
        codebook_loss = F.mse_loss(x_q, latent_embeddings.detach())

        quantization_loss = codebook_loss + self.mu * commitment_loss

        # preserve gradients
        x_q = latent_embeddings + (x_q - latent_embeddings).detach()

        indices = indices.view(latent_embeddings.shape[:-1])

        return x_q, quantization_loss, indices


class ResidualVectorQuantizer(nn.Module):
    def __init__(
            self, 
            num_codebooks, 
            codebook_size,
            embedding_dim,
        ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim

        self.vq_layers: list[VectorQuantizer] = nn.ModuleList([
            VectorQuantizer(codebook_size, embedding_dim) for _ in range(num_codebooks)
        ])

    def forward(self, latent_embeddings):
        all_losses = []
        all_indices = []

        x_q = 0
        residual = latent_embeddings

        for quantizer in self.vq_layers:
            x_res, loss, indices = quantizer(residual)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices


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
            cf_loss_weight=1.0,
            cf_embeddings=None
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
        self.cf_embeddings = cf_embeddings
        self.cf_loss_weight = cf_loss_weight

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

        self.rq = ResidualVectorQuantizer(
            num_codebooks=num_codebooks, 
            codebook_size=codebook_size,
            embedding_dim=embedding_dim
        )

    @staticmethod
    def get_codebook_indices(remainder, quantizer):
        dist = torch.sum(remainder**2, dim=1, keepdim=True) + torch.sum(quantizer.embedding.weight**2, dim=1, keepdim=True).t() - 2 * torch.matmul(remainder, quantizer.embedding.weight.t())
        return dist.argmin(dim=-1)

    def forward(self, inputs):
        latent_vector = self.encoder(inputs['embedding'])

        latent_restored = 0
        rqvae_loss = 0
        clusters = []
        remainder = latent_vector
        for quantizer in self.rq.vq_layers:
            codebook_indices = self.get_codebook_indices(remainder, quantizer)
            clusters.append(codebook_indices)

            quantized = quantizer.embedding(codebook_indices)
            codebook_vectors = remainder + (quantized - remainder).detach()

            rqvae_loss += self.beta * torch.nn.functional.mse_loss(remainder, quantized.detach())
            rqvae_loss += torch.nn.functional.mse_loss(quantized, remainder.detach())

            # codebook_vectors, quantizer_loss, codebook_indices = quantizer(remainder)
            # rqvae_loss += quantizer_loss

            latent_restored += codebook_vectors
            remainder = remainder - codebook_vectors
        
        embeddings_restored = self.decoder(latent_restored)
        recon_loss = F.mse_loss(embeddings_restored, inputs['embedding'])

        # TODO for now
        # if self.cf_embeddings is not None:
        #     cf_embedding_in_batch = self.cf_embeddings[item_ids]
        #     cf_embedding_in_batch = torch.from_numpy(cf_embedding_in_batch).to(quantized_embeddings.device)
        #     cf_loss = self.CF_loss(quantized_embeddings, cf_embedding_in_batch)
        # else:
        cf_loss = torch.as_tensor(0.0)

        loss = (recon_loss + self.quant_loss_weight * rqvae_loss + self.cf_loss_weight * cf_loss).mean()

        clusters_counts = []
        for cluster in clusters:
            clusters_counts.append(torch.bincount(cluster, minlength=self.codebook_size))

        # loss, recon_loss, cf_loss, rq_loss = self.compute_loss(
        #     content_embeddings=content_embeddings,
        #     out_embeddings=out_embeddings,
        #     item_ids=item_ids,
        #     rq_loss=rq_loss,
        #     quantized_embeddings=quantized_embeddings
        # )

        return loss, {
            'loss': loss.item(),
            'recon_loss': recon_loss.mean().item(),
            'rqvae_loss': rqvae_loss.mean().item(),
            'cf_loss': cf_loss.item(),

            'clusters_counts': clusters_counts,
            'clusters': torch.stack(clusters).T,
            'embedding_hat': embeddings_restored,
        }
    
    # def CF_loss(self, quantized_rep, encoded_rep):
    #     batch_size = quantized_rep.size(0)
    #     labels = torch.arange(batch_size, dtype=torch.long, device=quantized_rep.device)
    #     similarities = quantized_rep @ encoded_rep.T
    #     cf_loss = F.cross_entropy(similarities, labels)
    #     return cf_loss

    # @torch.no_grad()
    # def get_indices(self, content_embeddings):
    #     latent_embeddings = self.encoder(content_embeddings)
    #     _, _, indices = self.rq(latent_embeddings)
    #     return indices

    # def compute_loss(self, content_embeddings, out_embeddings, item_ids, rq_loss, quantized_embeddings):
    #     if self.loss_type == 'mse':
    #         recon_loss = F.mse_loss(content_embeddings, out_embeddings, reduction='mean')
    #     elif self.loss_type == 'l1':
    #         recon_loss = F.l1_loss(content_embeddings, out_embeddings, reduction='mean')
    #     else:
    #         raise ValueError('incompatible loss type')

    #     if self.cf_embeddings is not None:
    #         cf_embedding_in_batch = self.cf_embeddings[item_ids]
    #         cf_embedding_in_batch = torch.from_numpy(cf_embedding_in_batch).to(quantized_embeddings.device)
    #         cf_loss = self.CF_loss(quantized_embeddings, cf_embedding_in_batch)
    #     else:
    #         cf_loss = torch.as_tensor(0.0)

    #     total_loss = recon_loss + self.quant_loss_weight * rq_loss + self.cf_loss_weight * cf_loss

    #     return total_loss, recon_loss, cf_loss, rq_loss