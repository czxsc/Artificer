from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPModel, CLIPProcessor


@dataclass
class LabelVocab:
    artist_to_idx: dict[str, int]
    genre_to_idx: dict[str, int]
    style_to_idx: dict[str, int]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "LabelVocab":
        return cls(
            artist_to_idx={label: idx for idx, label in enumerate(sorted(frame["artist"].unique()))},
            genre_to_idx={label: idx for idx, label in enumerate(sorted(frame["genre"].unique()))},
            style_to_idx={label: idx for idx, label in enumerate(sorted(frame["style"].unique()))},
        )

    @classmethod
    def from_dict(cls, d: dict) -> "LabelVocab":
        return cls(
            artist_to_idx=d["artist_to_idx"],
            genre_to_idx=d["genre_to_idx"],
            style_to_idx=d["style_to_idx"],
        )

    @property
    def idx_to_artist(self) -> dict[int, str]:
        return {v: k for k, v in self.artist_to_idx.items()}

    @property
    def idx_to_genre(self) -> dict[int, str]:
        return {v: k for k, v in self.genre_to_idx.items()}

    @property
    def idx_to_style(self) -> dict[int, str]:
        return {v: k for k, v in self.style_to_idx.items()}


class ClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_type: str,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if head_type == "linear":
            self.layers = nn.Linear(input_dim, output_dim)
        else:
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class MultiHeadClipClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        vocab: LabelVocab,
        head_type: str,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = CLIPModel.from_pretrained(model_name)
        self.encoder.requires_grad_(False)
        embed_dim = self.encoder.config.projection_dim

        self.artist_head = ClassificationHead(
            embed_dim, len(vocab.artist_to_idx), head_type, hidden_dim, dropout
        )
        self.genre_head = ClassificationHead(
            embed_dim, len(vocab.genre_to_idx), head_type, hidden_dim, dropout
        )
        self.style_head = ClassificationHead(
            embed_dim, len(vocab.style_to_idx), head_type, hidden_dim, dropout
        )

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.encoder.vision_model(pixel_values=pixel_values)
        image_features = self.encoder.visual_projection(vision_outputs.pooler_output)
        return F.normalize(image_features, dim=-1)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        embeddings = self.encode_images(pixel_values)
        return {
            "artist": self.artist_head(embeddings),
            "genre": self.genre_head(embeddings),
            "style": self.style_head(embeddings),
        }


def load_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[MultiHeadClipClassifier, CLIPProcessor, LabelVocab]:
    """Load a trained classifier from a .pt checkpoint.

    Returns the model (eval mode), its processor, and the label vocab.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = LabelVocab.from_dict(checkpoint["label_vocab"])
    args = checkpoint["args"]

    model = MultiHeadClipClassifier(
        model_name=args["model_name"],
        vocab=vocab,
        head_type=args["head_type"],
        hidden_dim=args["hidden_dim"],
        dropout=args["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    processor = CLIPProcessor.from_pretrained(args["model_name"])
    return model, processor, vocab
