"""Shared image-retrieval engine (encoder-agnostic).

Path A of the project: embed images with a frozen vision encoder, then do cosine
top-k nearest-neighbor search against a database of known artworks. The database
stores image embeddings in one matrix and the artwork labels as a parallel metadata
table (joined back by position) -- labels are NOT embedded into the vectors.

Each concrete encoder (CLIP, DINOv2, SigLIP, ResNet/CNN) lives in its own file and
only implements how to turn a batch of images into a normalized embedding; the index,
search, and label-aggregation logic below are shared.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

LABEL_COLUMNS = ("artist", "genre", "style")


class BaseImageEncoder(ABC):
    """Wraps a frozen vision model and produces L2-normalized image embeddings.

    Subclasses set ``name``/``model_id`` and implement ``_load`` (populate
    ``self.model``, ``self.processor``, ``self.embed_dim``) and ``_extract`` (turn
    processor outputs into a [B, D] feature tensor before normalization).
    """

    name: str = "base"
    model_id: str = ""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim: int = 0
        self._load()
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    @abstractmethod
    def _load(self) -> None:
        """Populate self.model, self.processor, and self.embed_dim."""

    @abstractmethod
    def _extract(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return raw [B, D] features from the model's processed inputs."""

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        features = self._extract(inputs).float()
        return F.normalize(features, dim=-1)


@dataclass
class RetrievalIndex:
    """A searchable database: aligned (embeddings, metadata) with cosine search."""

    embeddings: np.ndarray  # [N, D] float32, L2-normalized
    metadata: pd.DataFrame  # N rows aligned to embeddings; has row_id + label columns
    encoder_name: str
    model_id: str

    @classmethod
    def build(
        cls,
        encoder: BaseImageEncoder,
        hf_dataset,
        manifest: pd.DataFrame,
        batch_size: int = 64,
        progress_every: int = 2000,
    ) -> "RetrievalIndex":
        rows = manifest.reset_index(drop=True)
        chunks: list[np.ndarray] = []
        for start in range(0, len(rows), batch_size):
            sub = rows.iloc[start : start + batch_size]
            images = [hf_dataset[int(rid)]["image"].convert("RGB") for rid in sub["row_id"]]
            embeddings = encoder.encode_images(images).cpu().numpy().astype("float32")
            chunks.append(embeddings)
            if progress_every and start % progress_every == 0:
                print(f"  encoded {start + len(sub)}/{len(rows)}", flush=True)
        stacked = (
            np.concatenate(chunks, axis=0)
            if chunks
            else np.zeros((0, encoder.embed_dim), dtype="float32")
        )
        return cls(stacked, rows, encoder.name, encoder.model_id)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "embeddings.npy", self.embeddings)
        self.metadata.to_csv(out_dir / "metadata.csv", index=False)
        (out_dir / "index_info.json").write_text(
            json.dumps(
                {
                    "encoder_name": self.encoder_name,
                    "model_id": self.model_id,
                    "num_items": int(self.embeddings.shape[0]),
                    "embed_dim": int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, out_dir: Path) -> "RetrievalIndex":
        embeddings = np.load(out_dir / "embeddings.npy")
        metadata = pd.read_csv(out_dir / "metadata.csv")
        info = json.loads((out_dir / "index_info.json").read_text(encoding="utf-8"))
        return cls(embeddings, metadata, info["encoder_name"], info["model_id"])

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores [Q, k], indices [Q, k]) sorted by descending cosine sim.

        Both query and stored embeddings are L2-normalized, so the dot product is the
        cosine similarity. Brute force is exact and fast at WikiArt scale (~40k items).
        """
        if query.ndim == 1:
            query = query[None, :]
        sims = query @ self.embeddings.T  # [Q, N]
        k = min(k, sims.shape[1])
        partitioned = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        partition_scores = np.take_along_axis(sims, partitioned, axis=1)
        order = np.argsort(-partition_scores, axis=1)
        indices = np.take_along_axis(partitioned, order, axis=1)
        scores = np.take_along_axis(partition_scores, order, axis=1)
        return scores, indices

    def aggregate_labels(
        self,
        indices: np.ndarray,
        scores: np.ndarray,
        method: str = "cosine_weighted",
        label_columns: tuple[str, ...] = LABEL_COLUMNS,
    ) -> list[dict[str, list[dict[str, float]]]]:
        """Vote on labels across the top-k neighbors of each query.

        method="cosine_weighted" weights each neighbor's vote by its similarity;
        method="flat" gives every neighbor an equal vote.
        Returns, per query, a dict of label_column -> ranked [{label, score}] list.
        """
        results: list[dict[str, list[dict[str, float]]]] = []
        for neighbor_idx, neighbor_scores in zip(indices, scores):
            if method == "cosine_weighted":
                weights = np.clip(neighbor_scores, a_min=0.0, a_max=None)
            else:
                weights = np.ones_like(neighbor_scores)

            per_column: dict[str, list[dict[str, float]]] = {}
            neighbor_rows = self.metadata.iloc[neighbor_idx]
            for column in label_columns:
                labels = neighbor_rows[column].to_numpy()
                votes: dict[str, float] = {}
                for label, weight in zip(labels, weights):
                    votes[label] = votes.get(label, 0.0) + float(weight)
                total = sum(votes.values()) or 1.0
                ranked = sorted(votes.items(), key=lambda item: item[1], reverse=True)
                per_column[column] = [
                    {"label": label, "score": weight / total} for label, weight in ranked
                ]
            results.append(per_column)
        return results
