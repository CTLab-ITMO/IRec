import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.utils import DEVICE
from models.base import BaseModel, TorchModel

# TODO finish tiger model
class TigerModel(TorchModel, config_name='tiger'):
    def __init__(
        self, 
        rqvae_encoder, 
        emb_dim, 
        n_tokens, 
        n_codebooks, 
        nhead, 
        num_encoder_layers, 
        num_decoder_layers, 
        dim_feedforward, 
        dropout
    ):
        super().__init__()
        
        self.rqvae_encoder = rqvae_encoder
        self.emb_dim = emb_dim
        self.n_tokens = n_tokens
        
        self.position_embeddings = nn.Embedding(n_codebooks, emb_dim)
        self.item_embeddings = nn.Embedding(n_tokens, emb_dim)
        
        self.transformer = nn.Transformer(
            d_model=emb_dim, 
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.proj = nn.Linear(emb_dim, n_tokens)
        
    @classmethod
    def create_from_config(cls, config, **kwargs):
        rqvae_train_config = json.load(open(config['rqvae_train_config_path']))
        
        rqvae_model = BaseModel.create_from_config(rqvae_train_config['model']).to(DEVICE)
        rqvae_model.load_state_dict(torch.load(config['rqvae_checkpoint_path'], weights_only=True))
        rqvae_model.eval()

        return cls(
            rqvae_encoder=rqvae_model,
            emb_dim=config['emb_dim'],
            n_tokens=config['n_tokens'],
            n_codebooks=config['n_codebooks'],
            nhead=config['nhead'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout']
        )    
    
    def forward(self, user_item_history):
        # Get item embeddings from RQVAE encoder
        item_sequence = self.rqvae_encoder(user_item_history)
        
        # Convert item sequence to embeddings (embedding size is emb_dim)
        item_embs = self.item_embeddings(item_sequence)
        
        # Add positional embeddings (positions are in the range [0, 3] for each tuple in the sequence)
        positions = torch.arange(0, item_embs.size(1), device=item_embs.device).unsqueeze(0)
        position_embs = self.position_embeddings(positions)
        
        # Add position embeddings to item embeddings
        embeddings = item_embs + position_embs
        
        # Transformer expects the input to be in (seq_len, batch, embedding_dim) format
        embeddings = embeddings.permute(1, 0, 2)  # Convert to (seq_len, batch, emb_dim)
        
        # Create the target sequence for the transformer decoder
        # You can shift the sequence for training as needed (e.g., teacher forcing)
        target = embeddings.clone()  # Use input embeddings as target for now
        
        # Pass through the transformer (using embeddings as both input and target)
        transformer_output = self.transformer(embeddings, target)
        
        # Project the output back to token space (256 possible values for each codebook)
        logits = self.proj(transformer_output)
        
        # Apply softmax to get probabilities (for cross-entropy loss)
        return logits

    def compute_loss(self, logits, target):
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits.view(-1, self.n_tokens), target.view(-1))
        return loss
