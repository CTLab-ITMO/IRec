import torch
import torch.nn as nn


def create_masked_tensor(data, lengths, is_right_aligned=False):
    batch_size = lengths.shape[0]
    max_sequence_length = lengths.max().item()

    if len(data.shape) == 1:  # only indices
        padded_tensor = torch.zeros(
            batch_size, max_sequence_length,
            dtype=data.dtype, device=data.device
        )  # (batch_size, max_seq_len)
    else:
        assert len(data.shape) == 2  # embeddings
        padded_tensor = torch.zeros(
            batch_size, max_sequence_length, data.shape[-1],
            dtype=data.dtype, device=data.device
        )  # (batch_size, max_seq_len, emb_dim)

    mask = torch.arange(
        end=max_sequence_length,
        device=data.device
    )[None].tile([batch_size, 1]) < lengths[:, None]  # (batch_size, max_seq_len)

    if is_right_aligned:
        mask = torch.flip(mask, dims=[-1])
    padded_tensor[mask] = data

    return padded_tensor, mask


class TorchModel(nn.Module):
    @torch.no_grad()
    def _init_weights(self, initializer_range):
        for key, value in self.named_parameters():
            if 'weight' in key:
                if 'norm' in key:
                    nn.init.ones_(value.data)
                else:
                    nn.init.trunc_normal_(
                        value.data,
                        std=initializer_range,
                        a=-2 * initializer_range,
                        b=2 * initializer_range
                    )
            elif 'bias' in key:
                nn.init.zeros_(value.data)
            elif 'codebook' in key:
                nn.init.trunc_normal_(
                    value.data,
                    std=initializer_range,
                    a=-2 * initializer_range,
                    b=2 * initializer_range
                )
            elif 'bos_embedding' in key:
                nn.init.trunc_normal_(
                    value.data,
                    std=initializer_range,
                    a=-2 * initializer_range,
                    b=2 * initializer_range,
                )
            else:
                raise ValueError(f'Unknown transformer weight: {key}')


class AutoCast(nn.Module):
    def __init__(self, module, dtype=torch.bfloat16, device_type='cuda', cache_enabled=True):
        super().__init__()
        self.module = module
        self._dtype = dtype
        self._device_type = device_type
        self._cache_enabled = cache_enabled

    @property
    def dtype(self):
        return self._dtype

    @property
    def cache_enabled(self):
        return self._cache_enabled

    def forward(self, *args, **kwargs):
        with torch.autocast(
            device_type=self._device_type,
            dtype=self._dtype,
            cache_enabled=self._cache_enabled
        ):
            return self.module(*args, **kwargs)
