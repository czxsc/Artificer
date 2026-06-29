from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

from data.wikiart_utils import load_wikiart_dataset


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "phase1"
DEFAULT_SPLIT_DIR = OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frozen-CLIP multi-head classifier for WikiArt metadata."
    )
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--head-type",
        choices=("linear", "mlp"),
        default="mlp",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


class WikiArtSplitDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        manifest: pd.DataFrame,
        processor: CLIPProcessor,
        vocab: LabelVocab,
    ) -> None:
        self.hf_dataset = hf_dataset
        self.manifest = manifest.reset_index(drop=True)
        self.processor = processor
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.manifest.iloc[index]
        sample = self.hf_dataset[int(row["row_id"])]
        image = sample["image"].convert("RGB")

        encoded = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "artist": torch.tensor(self.vocab.artist_to_idx[row["artist"]], dtype=torch.long),
            "genre": torch.tensor(self.vocab.genre_to_idx[row["genre"]], dtype=torch.long),
            "style": torch.tensor(self.vocab.style_to_idx[row["style"]], dtype=torch.long),
        }


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
        image_features = self.encoder.get_image_features(pixel_values=pixel_values)
        return F.normalize(image_features, dim=-1)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        embeddings = self.encode_images(pixel_values)
        return {
            "artist": self.artist_head(embeddings),
            "genre": self.genre_head(embeddings),
            "style": self.style_head(embeddings),
        }


def load_manifest(split_dir: Path, name: str, limit: int | None) -> pd.DataFrame:
    frame = pd.read_csv(split_dir / f"{name}_manifest.csv")
    if limit is not None:
        frame = frame.head(limit).copy()
    return frame


def make_loader(
    hf_dataset,
    frame: pd.DataFrame,
    processor: CLIPProcessor,
    vocab: LabelVocab,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = WikiArtSplitDataset(hf_dataset, frame, processor, vocab)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    top1 = logits.argmax(dim=1).cpu().numpy()
    y_true = labels.cpu().numpy()
    topk = min(5, logits.shape[1])
    topk_indices = logits.topk(k=topk, dim=1).indices.cpu().numpy()
    topk_accuracy = float(np.mean([truth in preds for truth, preds in zip(y_true, topk_indices)]))
    accuracy = float(np.mean(top1 == y_true))

    macro_f1_scores: list[float] = []
    for class_index in np.unique(y_true):
        true_positive = np.sum((top1 == class_index) & (y_true == class_index))
        false_positive = np.sum((top1 == class_index) & (y_true != class_index))
        false_negative = np.sum((top1 != class_index) & (y_true == class_index))

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator if precision_denominator else 0.0
        )
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        if precision + recall == 0:
            macro_f1_scores.append(0.0)
        else:
            macro_f1_scores.append(2 * precision * recall / (precision + recall))

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(macro_f1_scores)),
        "top5_accuracy": topk_accuracy,
    }


def evaluate(
    model: MultiHeadClipClassifier,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    model.eval()
    collected: dict[str, list[torch.Tensor]] = {
        "artist_logits": [],
        "genre_logits": [],
        "style_logits": [],
        "artist_labels": [],
        "genre_labels": [],
        "style_labels": [],
    }

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            outputs = model(pixel_values)
            for label_name in ("artist", "genre", "style"):
                collected[f"{label_name}_logits"].append(outputs[label_name].cpu())
                collected[f"{label_name}_labels"].append(batch[label_name].cpu())

    metrics: dict[str, dict[str, float]] = {}
    for label_name in ("artist", "genre", "style"):
        logits = torch.cat(collected[f"{label_name}_logits"], dim=0)
        labels = torch.cat(collected[f"{label_name}_labels"], dim=0)
        metrics[label_name] = compute_metrics(logits, labels)

    return metrics


def train_one_epoch(
    model: MultiHeadClipClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        artist = batch["artist"].to(device)
        genre = batch["genre"].to(device)
        style = batch["style"].to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values)
        loss = (
            F.cross_entropy(outputs["artist"], artist)
            + F.cross_entropy(outputs["genre"], genre)
            + F.cross_entropy(outputs["style"], style)
        ) / 3.0
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * pixel_values.size(0)

    return running_loss / len(dataloader.dataset)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_frame = load_manifest(args.split_dir, "train", args.max_train_samples)
    val_frame = load_manifest(args.split_dir, "val", args.max_eval_samples)
    test_frame = load_manifest(args.split_dir, "test", args.max_eval_samples)
    vocab = LabelVocab.from_frame(train_frame)

    processor = CLIPProcessor.from_pretrained(args.model_name)
    hf_dataset = load_wikiart_dataset()

    train_loader = make_loader(
        hf_dataset, train_frame, processor, vocab, args.batch_size, True, args.num_workers
    )
    val_loader = make_loader(
        hf_dataset, val_frame, processor, vocab, args.batch_size, False, args.num_workers
    )
    test_loader = make_loader(
        hf_dataset, test_frame, processor, vocab, args.batch_size, False, args.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHeadClipClassifier(
        model_name=args.model_name,
        vocab=vocab,
        head_type=args.head_type,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )

    history: list[dict[str, object]] = []
    best_val_macro_f1 = -1.0
    best_state_dict = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        mean_val_macro_f1 = float(
            np.mean([val_metrics[label]["macro_f1"] for label in ("artist", "genre", "style")])
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_metrics": val_metrics,
            }
        )
        print(json.dumps(history[-1], indent=2))

        if mean_val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = mean_val_macro_f1
            best_state_dict = {key: value.cpu() for key, value in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_metrics = evaluate(model, test_loader, device)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_vocab": {
                "artist_to_idx": vocab.artist_to_idx,
                "genre_to_idx": vocab.genre_to_idx,
                "style_to_idx": vocab.style_to_idx,
            },
            "args": vars(args),
        },
        OUTPUT_DIR / "clip_classifier.pt",
    )
    (OUTPUT_DIR / "clip_classifier_metrics.json").write_text(
        json.dumps({"history": history, "test_metrics": test_metrics}, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({"test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
