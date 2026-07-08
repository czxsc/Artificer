"""DINOv2 image encoder for retrieval.

Self-supervised ViT with no text alignment. Tends to excel at exact / instance-level
matching (retrieving the same painting even when blurred, cropped, or tilted) because
it captures fine visual detail rather than semantics. Uses the CLS pooler output.
"""
from __future__ import annotations

import torch
from transformers import AutoImageProcessor, AutoModel

from ml.models.retrieval.index import BaseImageEncoder


class DinoEncoder(BaseImageEncoder):
    name = "dino"

    def __init__(
        self,
        model_id: str = "facebook/dinov2-base",
        device: torch.device | None = None,
    ) -> None:
        self.model_id = model_id
        super().__init__(device)

    def _load(self) -> None:
        self.model = AutoModel.from_pretrained(self.model_id)
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.embed_dim = self.model.config.hidden_size

    def _extract(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(pixel_values=inputs["pixel_values"])
        return outputs.pooler_output
