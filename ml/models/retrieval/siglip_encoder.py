"""SigLIP image encoder for retrieval.

A CLIP-style image/text model (sigmoid contrastive loss) that is often stronger than
CLIP at retrieval. Uses ``get_image_features`` for the pooled image embedding.
"""
from __future__ import annotations

import torch
from transformers import AutoImageProcessor, SiglipVisionModel

from ml.models.retrieval.index import BaseImageEncoder


class SiglipEncoder(BaseImageEncoder):
    name = "siglip"

    def __init__(
        self,
        model_id: str = "google/siglip-base-patch16-224",
        device: torch.device | None = None,
    ) -> None:
        self.model_id = model_id
        super().__init__(device)

    def _load(self) -> None:
        # Vision tower only: skips SigLIP's text weights and its sentencepiece
        # tokenizer, neither of which we need for image embeddings.
        self.model = SiglipVisionModel.from_pretrained(self.model_id)
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.embed_dim = self.model.config.hidden_size

    def _extract(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(pixel_values=inputs["pixel_values"])
        return outputs.pooler_output
