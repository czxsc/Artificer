"""Predict top-k artist/genre/style labels for an image using a trained CLIP classifier.

Usage (after training):
    python scripts/predict_labels.py path/to/image.jpg
    python scripts/predict_labels.py path/to/image.jpg --top-k 3
    python scripts/predict_labels.py path/to/image.jpg --checkpoint outputs/phase1/clip_classifier.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from transformers import CLIPProcessor

from ml.models.clip.classifier import LabelVocab, MultiHeadClipClassifier, load_from_checkpoint


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1] / "outputs" / "phase1" / "clip_classifier.pt"
)
LABEL_TYPES = ("artist", "genre", "style")


def predict_top_k(
    model: MultiHeadClipClassifier,
    processor: CLIPProcessor,
    vocab: LabelVocab,
    image: Image.Image,
    device: torch.device,
    k: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Run the classifier on a single PIL image and return top-k predictions per label type.

    Returns:
        {
          "artist": [{"label": "claude-monet", "score": 0.92}, ...],
          "genre":  [...],
          "style":  [...],
        }
    """
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values)

    idx_to_label = {
        "artist": vocab.idx_to_artist,
        "genre": vocab.idx_to_genre,
        "style": vocab.idx_to_style,
    }

    result: dict[str, list[dict[str, Any]]] = {}
    for label_type in LABEL_TYPES:
        logits = outputs[label_type][0]
        probs = torch.softmax(logits, dim=0)
        top_probs, top_indices = probs.topk(min(k, len(probs)))
        result[label_type] = [
            {"label": idx_to_label[label_type][idx.item()], "score": round(prob.item(), 4)}
            for prob, idx in zip(top_probs, top_indices)
        ]

    return result


def predict_labels_for_image(
    image_path: str | Path,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    top_k: int = 5,
    device: torch.device | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Convenience function: load checkpoint + image, return top-k labels.

    Useful as a library call from other scripts or the RAG pipeline.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, processor, vocab = load_from_checkpoint(checkpoint_path, device)
    image = Image.open(image_path).convert("RGB")
    return predict_top_k(model, processor, vocab, image, device, k=top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict art labels for an image.")
    parser.add_argument("image_path", type=Path, help="Path to input image file")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to trained .pt checkpoint (default: outputs/phase1/clip_classifier.pt)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions per label")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Run training first: python scripts/train_clip_classifier.py",
            file=sys.stderr,
        )
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {args.checkpoint} on {device}...", file=sys.stderr)
    model, processor, vocab = load_from_checkpoint(args.checkpoint, device)

    image = Image.open(args.image_path).convert("RGB")
    predictions = predict_top_k(model, processor, vocab, image, device, k=args.top_k)

    print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()
