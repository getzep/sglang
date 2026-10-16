import torch

from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator


class ClefPoolConfigurator(DefaultPoolConfigurator):
    def __init__(self, kvc):
        super().__init__(kvc)
        width = kvc.model_config.joint_head_config["width"]
        self._clef_memory_bytes_per_token = width * torch._utils._element_size(
            kvc.model_config.dtype
        )
        self._cell_size += self._clef_memory_bytes_per_token

    def calculate_pool_sizes(self, available_bytes: int, page_size: int):
        return super().calculate_pool_sizes(
            available_bytes - self._clef_memory_bytes_per_token * page_size,
            page_size,
        )
