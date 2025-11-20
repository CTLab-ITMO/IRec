import torch
from typing import Dict, Any

from irec.callbacks.base import BatchCallback
from irec.runners.base import BatchRunner, BatchRunnerContext


class Thermometer(BatchCallback):
    def __init__(self, stats=None, **modules: Dict[str, torch.nn.Module]):
        super().__init__()
        for module in modules.values():
            assert isinstance(module, torch.nn.Module)
        self.modules = modules
        self.hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self.stats = stats or ['max', 'min', 'mean', 'median', 'std']

    def before_batch(self, runner: BatchRunner, context: BatchRunnerContext) -> None:
        self._register_all(context)

    def after_step(self, runner: BatchRunner, context: BatchRunnerContext) -> None:
        self._remove_all()

    def _register_all(self, context: BatchRunnerContext) -> None:
        for name, layer in self.modules.items():
            if name not in self.hooks:
                self.hooks[name] = layer.register_forward_hook(
                    self._make_hook(name, context)
                )

    def _remove_all(self) -> None:
        for name, handle in list(self.hooks.items()):
            handle.remove()
            del self.hooks[name]

    def _calculate_stats(self, tensor: torch.Tensor, layer_name: str, context: Any) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

        for stat in self.stats:
            if callable(stat):
                value = stat(tensor)
                stat_name = stat.__name__
            elif stat == 'max':
                value = tensor.amax().detach()
                stat_name = 'max'
            elif stat == 'min':
                value = tensor.amin().detach()
                stat_name = 'min'
            elif stat == 'mean':
                value = tensor.mean().detach()
                stat_name = 'mean'
            elif stat == 'median':
                value = tensor.median().detach()
                stat_name = 'median'
            elif stat == 'std':
                value = tensor.std().detach()
                stat_name = 'std'
            else:
                raise ValueError(f"Unknown statistic: {stat}")

            context.metrics[f"act/{layer_name}_{stat_name}"] = value

    def _make_hook(self, layer_name: str, context: Any):
        def hook(_, __, out):
            if isinstance(out, torch.Tensor):
                tensor = out
            elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                tensor = out[0]
            else:
                return
            self._calculate_stats(tensor, layer_name, context)

        return hook
