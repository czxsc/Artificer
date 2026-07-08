"""CLIP image encoder for retrieval (Path A baseline).

Mirrors the projection used by the CLIP classifier's ``encode_images`` (vision pooler
-> visual_projection -> L2 normalize) so retrieval and classification share the exact
same 512-d embedding, keeping the Path A vs Path B comparison fair.
"""
from __future__ import annotations

import torch
from transformers import CLIPModel, CLIPProcessor

from ml.models.retrieval.index import BaseImageEncoder


class ClipEncoder(BaseImageEncoder):
    name = "clip"

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        device: torch.device | None = None,
    ) -> None:
        self.model_id = model_id
        super().__init__(device)

    def _load(self) -> None:
        self.model = CLIPModel.from_pretrained(self.model_id)
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.embed_dim = self.model.config.projection_dim

    def _extract(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        vision_outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
        return self.model.visual_projection(vision_outputs.pooler_output)
