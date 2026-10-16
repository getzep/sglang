import torch

from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool


class ClefHybridLinearKVPool(HybridLinearKVPool):
    def __init__(self, *args, clef_width: int, clef_dtype: torch.dtype, **kwargs):
        super().__init__(*args, **kwargs)
        self.clef_memory = torch.empty(
            (self.size + self.page_size, clef_width),
            dtype=clef_dtype,
            device=self.device,
        )

    def store_clef_memory(self, locations: torch.Tensor, rows: torch.Tensor) -> None:
        self.clef_memory[locations] = rows

    def gather_clef_memory(self, locations: torch.Tensor) -> torch.Tensor:
        return self.clef_memory[locations]


__all__ = ["ClefHybridLinearKVPool"]
