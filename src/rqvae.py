import torch

from sklearn.cluster import KMeans

class RQVAE(torch.nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            beta: float,
            codebook_sizes: list[int],
            should_init_codebooks=False,
            should_reinit_unused_clusters=False
        ):
        super().__init__()

        # In original paper it is set to 0.25
        self.register_buffer('beta', torch.tensor(beta))  

        # Kmeans initialization
        self.should_init_codebooks = should_init_codebooks

        # Trick with re-initing empty clusters
        self.should_reinit_unused_clusters = should_reinit_unused_clusters 

        self.mse_loss = torch.nn.MSELoss()

        # Enc and dec are mirrored copies of each other
        self.encoder = self.make_encoding_tower(input_dim, hidden_dim)
        self.decoder = self.make_encoding_tower(hidden_dim, input_dim)

        # Default initialization of codebook
        self.codebooks = torch.nn.ParameterList()
        for codebook_size in codebook_sizes:
            cb = torch.FloatTensor(codebook_size, hidden_dim)
            with torch.no_grad():
                torch.nn.init.trunc_normal_(cb, std=0.02, a=-2 * 0.02, b=2 * 0.02)
            self.codebooks.append(cb)

    def make_encoding_tower(self, d1: int, d2: int):
        return torch.nn.Linear(d1, d2, bias=False)
    
    # Get closest index for given embedding
    @staticmethod
    def get_codebook_indices(remainder, codebook):
        dist = torch.cdist(remainder, codebook)
        return dist.argmin(dim=-1)

    # Recursive k-means initialization
    @staticmethod
    def kmeans(embeddings, num_clusters, num_steps=300):
        # Just dummy kmeans implementation to get some better initial point
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1) # ???
        closest_cluster = torch.randint(0, num_clusters, (embeddings.shape[0], ), device=embeddings.device)
        cluster_centers = torch.zeros((num_clusters, embeddings.shape[1]), device=embeddings.device)
        for clust_ind in range(num_clusters):
            print(clust_ind, (closest_cluster == clust_ind).sum())
            cluster_centers[clust_ind] = embeddings[closest_cluster == clust_ind].mean(dim=0)

        for iter in range(num_steps):
            dist = torch.cdist(embeddings, cluster_centers)
            closest_cluster = dist.argmin(dim=-1)
            print('Kmeans iter:', iter, closest_cluster.shape)
            for clust_ind in range(num_clusters):
                if clust_ind == 0:
                    print(clust_ind, (closest_cluster == clust_ind).sum())
                cluster_centers[clust_ind] = embeddings[closest_cluster == clust_ind].mean(dim=0)

        return cluster_centers

    def init_codebooks(self, embeddings):
        with torch.no_grad():
            remainder = self.encoder(embeddings)
            for codebook in self.codebooks:
                codebook.data = self.kmeans(embeddings=remainder, num_clusters=codebook.shape[0])
                codebook_indices = self.get_codebook_indices(remainder, codebook)
                codebook_vectors = codebook[codebook_indices]
                remainder = remainder - codebook_vectors

    @staticmethod
    def reinit_unused_clusters(remainder, codebook, codebook_indices):
        with torch.no_grad():
            is_used = torch.full((codebook.shape[0], ), False, device=codebook.device)
            unique_indices = codebook_indices.unique()
            is_used[unique_indices] = True
            rand_input = torch.randint(0, remainder.shape[0], ((~is_used).sum(), ))
            codebook[~is_used] = remainder[rand_input]

    def forward(self, inputs):
        embeddings = inputs['embedding']
        if self.should_init_codebooks:
            self.init_codebooks(embeddings)
            self.should_init_codebooks = False
        latent_vector = self.encoder(embeddings)

        latent_restored = 0
        rqvae_loss = 0
        num_unique_clusters = []
        remainder = latent_vector
        for codebook in self.codebooks:
            codebook_indices = self.get_codebook_indices(remainder, codebook)
            codebook_vectors = codebook[codebook_indices]

            if self.should_reinit_unused_clusters:
                self.reinit_unused_clusters(remainder, codebook, codebook_indices)

            num_unique_clusters.append(codebook_indices.unique().shape[0])
            rqvae_loss += self.beta * self.mse_loss(remainder, codebook_vectors.detach())
            rqvae_loss += self.mse_loss(codebook_vectors, remainder.detach())

            latent_restored = latent_restored + codebook_vectors
            remainder = remainder - codebook_vectors

        # Here we cast recon loss to latent vector
        latent_restored = latent_vector + (latent_restored - latent_vector).detach()
        embeddings_restored = self.decoder(latent_restored)
        recon_loss = self.mse_loss(embeddings_restored, embeddings)
        loss = (recon_loss + rqvae_loss).mean()

        return {
            'loss': loss,
            'recon_loss': recon_loss.mean().detach(),
            'rqvae_loss': rqvae_loss.mean().detach(),
            **{
                f'unique/{i}': cnt
                for i, cnt in enumerate(num_unique_clusters)
            }
        }