from typing import Any, Callable, Optional, Self, Union

import einops
import torch
import torch.nn.functional as F

from flash_attn.modules.mha import FlashSelfAttention


class MHAttention(torch.nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            num_heads: Optional[int] = None,
            dropout: float = 0.0,
            window_size: tuple[int, int] = (-1, -1),
            return_residual: bool = True,
            causal: bool = False,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.window_size = window_size
        self.return_residual = return_residual
        self.causal = causal

        self.num_heads = num_heads or embedding_dim // 64
        assert self.embedding_dim % self.num_heads == 0
        self.head_dim = self.embedding_dim // self.num_heads

        self.self_attention = FlashSelfAttention(
            causal=self.causal,
            attention_dropout=self.dropout,
            window_size=self.window_size,
        )

        self.Wqkv = torch.nn.Linear(self.embedding_dim, 3 * self.head_dim * self.num_heads)

        self.out_proj = torch.nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(
            self,
            x: torch.Tensor,
            lengths: torch.Tensor,
            cu_seqlens: torch.Tensor,
            max_seqlen: torch.Tensor,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        qkv = self.Wqkv(x)

        qkv = einops.rearrange(qkv, '... (three h d) -> ... three h d', three=3, d=self.head_dim)

        result = self.self_attention(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        result = self.out_proj(einops.rearrange(result, '... h d -> ... (h d)'))

        return (result, x) if self.return_residual else result


class MLP(torch.nn.Module):
    ACTIVATIONS = {'relu': F.relu, 'sigmoid': F.sigmoid, 'gelu': F.gelu, 'swiglu': F.silu}

    def __init__(
            self,
            in_features: int,
            dropout: float = 0.0,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
            bias1: bool = True,
            bias2: bool = True,
            return_residual: bool = False,
    ):
        super().__init__()

        self.activation = MLP.ACTIVATIONS[activation] if isinstance(activation, str) else activation

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.return_residual = return_residual

        self.fc1 = torch.nn.Linear(in_features, hidden_features, bias=bias1)
        self.fc2 = torch.nn.Linear(hidden_features, out_features, bias=bias2)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        y = self.fc1(x)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.fc2(y)
        return (y, x) if self.return_residual else y


class Block(torch.nn.Module):
    NORMALIZATIONS = {'layer_norm': torch.nn.LayerNorm, 'rms_norm': torch.nn.RMSNorm}

    def __init__(
            self,
            mixer: torch.nn.Module,
            mlp: torch.nn.Module,
            dropout1: torch.nn.Module,
            norm1: torch.nn.Module,
            norm2: torch.nn.Module,
    ):
        super().__init__()

        self.mixer = mixer
        self.mlp = mlp

        self.dropout1 = dropout1
        self.norm1 = norm1
        self.norm2 = norm2

    def forward(
            self,
            hidden_states: torch.Tensor,
            mixer_kwargs: Optional[dict[str, Any]] = None,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if mixer_kwargs is None:
            mixer_kwargs = {}

        mixed = self.mixer(hidden_states, **mixer_kwargs)
        dropped = self.dropout1(mixed)
        residual = hidden_states + dropped
        hidden_states = self.norm1(residual)

        mlped = self.mlp(hidden_states)
        hidden_states = hidden_states + mlped
        hidden_states = self.norm2(residual)
        
        return hidden_states

    @staticmethod
    def make_normalization(norm: str, **kwargs):
        return Block.NORMALIZATIONS[norm](kwargs['embedding_dim'], eps=kwargs['eps'])

    @classmethod
    def make_default(
            cls: type[Self],
            embedding_dim: int,
            dim_feedforward: int,
            num_heads: Optional[int] = None,
            dropout: float = 0.0,
            norm: str = 'layer_norm',
            activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
            causal: bool = False,
            eps: float = 1e-5,
            attn_dropout: float = 0.0,
            window_size: tuple[int, int] = (-1, -1),
    ) -> Self:

        norm1 = Block.make_normalization(norm, embedding_dim=embedding_dim, eps=eps)
        norm2 = Block.make_normalization(norm, embedding_dim=embedding_dim, eps=eps)

        mixer = MHAttention(
            embedding_dim=embedding_dim,
            dropout=attn_dropout,
            return_residual=False,
            causal=causal,
            num_heads=num_heads,
            window_size=window_size,
        )

        mlp = MLP(in_features=embedding_dim, dropout=dropout, hidden_features=dim_feedforward, return_residual=False, activation=activation)

        block = cls(
            mixer=mixer,
            mlp=mlp,
            dropout1=torch.nn.Dropout(dropout),
            norm1=norm1,
            norm2=norm2,
        )

        return block


class TransformerEncoder(torch.nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            dim_feedforward: int,
            layers: Union[int, list[Block]],
            num_heads: Optional[int] = None,
            dropout: Union[float, tuple[float, float]] = 0.0,
            norm: str = 'layer_norm',
            activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
            causal: bool = False,
            attn_dropout: float = 0.0,
            window_size: tuple[int, int] = (-1, -1),
    ):
        super().__init__()

        if isinstance(layers, int):
            layers = [
                Block.make_default(
                    embedding_dim=embedding_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    norm=norm,
                    causal=causal,
                    activation=activation,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    window_size=window_size,
                )
                for _ in range(layers)
            ]

        self._embedding_dim = embedding_dim
        self.layers = torch.nn.ModuleList(layers)

    @property
    def embedding_dim(self):
        return self._embedding_dim

    def forward(
            self,
            embeddings: torch.Tensor,
            lengths: torch.Tensor,
            max_seqlen: Optional[int] = None,
            **mixer_kwargs
    ) -> torch.Tensor:
        cu_seqlens = F.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0))

        if max_seqlen is None:
            max_seqlen = lengths.max().item()

        mixer_kwargs.update({'cu_seqlens': cu_seqlens, 'max_seqlen': max_seqlen, 'lengths': lengths})

        for layer in self.layers:
            embeddings = layer(embeddings, mixer_kwargs=mixer_kwargs)

        return embeddings
