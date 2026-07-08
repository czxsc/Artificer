"""CNN (ResNet-50) image encoder for retrieval.

A non-transformer, ImageNet-pretrained convolutional baseline. Uses the global
average-pooled feature map (2048-d) as the embedding. Loaded via transformers'
``ResNetModel`` so no torchvision/timm dependency is required.
"""
from __future__ import annotations

import torch
from transformers import AutoImageProcessor, AutoModel

from ml.models.retrieval.index import BaseImageEncoder


class CnnEncoder(BaseImageEncoder):
    name = "cnn"

    def __init__(
        self,
        model_id: str = "microsoft/resnet-50",
        device: torch.device | None = None,
    ) -> None:
        self.model_id = model_id
        super().__init__(device)

    def _load(self) -> None:
        self.model = AutoModel.from_pretrained(self.model_id)
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        # ResNet config exposes stage widths; the last one is the pooled feature dim.
        self.embed_dim = self.model.config.hidden_sizes[-1]

    def _extract(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(pixel_values=inputs["pixel_values"])
        # pooler_output is [B, C, 1, 1] for ResNet -> flatten to [B, C].
        return outputs.pooler_output.flatten(1)
