"""Image-retrieval (Path A) package: a shared engine + swappable vision encoders."""
from __future__ import annotations

import torch

from ml.models.retrieval.clip_encoder import ClipEncoder
from ml.models.retrieval.cnn_encoder import CnnEncoder
from ml.models.retrieval.dino_encoder import DinoEncoder
from ml.models.retrieval.index import LABEL_COLUMNS, BaseImageEncoder, RetrievalIndex
from ml.models.retrieval.siglip_encoder import SiglipEncoder

ENCODERS: dict[str, type[BaseImageEncoder]] = {
    "clip": ClipEncoder,
    "dino": DinoEncoder,
    "siglip": SiglipEncoder,
    "cnn": CnnEncoder,
}


def get_encoder(name: str, device: torch.device | None = None, **kwargs) -> BaseImageEncoder:
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder '{name}'. Options: {sorted(ENCODERS)}")
    return ENCODERS[name](device=device, **kwargs)


__all__ = [
    "BaseImageEncoder",
    "RetrievalIndex",
    "LABEL_COLUMNS",
    "ClipEncoder",
    "DinoEncoder",
    "SiglipEncoder",
    "CnnEncoder",
    "ENCODERS",
    "get_encoder",
]
