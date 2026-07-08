from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# On Windows, importing `datasets` (which loads pyarrow's native libs) AFTER torch
# triggers a DLL load-order access violation with the CUDA torch build. Import it
# first so pyarrow loads before torch. Do NOT move this below the torch import.
import datasets  # noqa: F401  (import-order side effect; used via data.wikiart_utils)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import CLIPProcessor

try:
    import matplotlib.pyplot as plt  # type: ignore
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
from data.wikiart_utils import load_wikiart_dataset
from ml.models.clip.classifier import ClassificationHead, LabelVocab, MultiHeadClipClassifier


REPO_ROOT = Path(__file__).resolve().parents[1]
# Trained model checkpoints + metrics live under phase1 (the model-experiment phase).
OUTPUT_DIR = REPO_ROOT / "outputs" / "phase1"
# Dataset splits are the canonical phase0 manifests (Unknown Artist removed).
DEFAULT_SPLIT_DIR = REPO_ROOT / "outputs" / "phase0"
TEST_MANIFEST_NAME = "known_artworks_test"


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
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping: stop if val mean macro-F1 has not improved for this many "
        "epochs. Set to 0 to disable early stopping.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.005,
        help="Minimum val macro-F1 gain to count as an improvement for early stopping.",
    )
    parser.add_argument(
        "--early-stop-metric",
        choices=("mean", "artist", "genre", "style"),
        default="artist",
        help="Which val macro-F1 to watch for early stopping. Default 'artist': the artist "
        "head keeps improving after genre/style plateau, so watching the mean stops too early.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Recompute CLIP embeddings even if a cache exists.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_split(
    model: MultiHeadClipClassifier,
    processor: CLIPProcessor,
    hf_dataset,
    frame: pd.DataFrame,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Run the frozen CLIP encoder over a split's images once, returning [N, D]."""
    model.eval()
    row_ids = frame["row_id"].tolist()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(row_ids), batch_size):
            batch_ids = row_ids[start : start + batch_size]
            images = [hf_dataset[int(rid)]["image"].convert("RGB") for rid in batch_ids]
            pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            embeddings = model.encode_images(pixel_values).cpu().numpy().astype("float32")
            chunks.append(embeddings)
            if start % 3200 == 0:
                print(f"    encoded {start + len(batch_ids)}/{len(row_ids)}", flush=True)
    return np.concatenate(chunks, axis=0)


def get_embeddings(
    model: MultiHeadClipClassifier,
    processor: CLIPProcessor,
    hf_dataset,
    frame: pd.DataFrame,
    split_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    """Return a split's embeddings, computing + caching them to disk on first use.

    The frozen encoder makes embeddings stable across runs, so they are cached keyed by
    (model, exact set of row_ids). Pass --refresh-cache (or delete
    outputs/phase1/emb_cache) to force a recompute.
    """
    cache_dir = OUTPUT_DIR / "emb_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model_name.replace("/", "_")
    emb_path = cache_dir / f"{slug}__{split_name}.npy"
    meta_path = cache_dir / f"{slug}__{split_name}.json"

    row_ids = frame["row_id"].tolist()
    key = hashlib.md5(
        (args.model_name + "|" + ",".join(str(rid) for rid in row_ids)).encode()
    ).hexdigest()

    if not args.refresh_cache and emb_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("key") == key:
            print(f"  [{split_name}] loaded {len(row_ids)} cached embeddings.", flush=True)
            return np.load(emb_path)

    print(f"  [{split_name}] encoding {len(row_ids)} images with {args.model_name}...", flush=True)
    embeddings = encode_split(model, processor, hf_dataset, frame, args.batch_size, device)
    np.save(emb_path, embeddings)
    meta_path.write_text(
        json.dumps({"key": key, "model": args.model_name, "count": len(row_ids)}),
        encoding="utf-8",
    )
    return embeddings


def load_manifest(split_dir: Path, name: str, limit: int | None) -> pd.DataFrame:
    frame = pd.read_csv(split_dir / f"{name}_manifest.csv")
    if limit is not None:
        frame = frame.head(limit).copy()
    return frame


def label_index_tensors(
    frame: pd.DataFrame, vocab: LabelVocab
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    artist = torch.tensor([vocab.artist_to_idx[x] for x in frame["artist"]], dtype=torch.long)
    genre = torch.tensor([vocab.genre_to_idx[x] for x in frame["genre"]], dtype=torch.long)
    style = torch.tensor([vocab.style_to_idx[x] for x in frame["style"]], dtype=torch.long)
    return artist, genre, style


def make_embedding_loader(
    embeddings: np.ndarray,
    frame: pd.DataFrame,
    vocab: LabelVocab,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    artist, genre, style = label_index_tensors(frame, vocab)
    dataset = TensorDataset(torch.from_numpy(embeddings), artist, genre, style)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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
    label_names = ("artist", "genre", "style")
    collected: dict[str, list[torch.Tensor]] = {f"{n}_logits": [] for n in label_names}
    collected.update({f"{n}_labels": [] for n in label_names})

    with torch.no_grad():
        for embeddings, artist, genre, style in dataloader:
            outputs = model.classify_embeddings(embeddings.to(device))
            labels = {"artist": artist, "genre": genre, "style": style}
            for label_name in label_names:
                collected[f"{label_name}_logits"].append(outputs[label_name].cpu())
                collected[f"{label_name}_labels"].append(labels[label_name])

    metrics: dict[str, dict[str, float]] = {}
    for label_name in label_names:
        logits = torch.cat(collected[f"{label_name}_logits"], dim=0)
        labels_cat = torch.cat(collected[f"{label_name}_labels"], dim=0)
        metrics[label_name] = compute_metrics(logits, labels_cat)

    return metrics


def train_one_epoch(
    model: MultiHeadClipClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for embeddings, artist, genre, style in dataloader:
        embeddings = embeddings.to(device)
        artist = artist.to(device)
        genre = genre.to(device)
        style = style.to(device)

        optimizer.zero_grad()
        outputs = model.classify_embeddings(embeddings)
        loss = (
            F.cross_entropy(outputs["artist"], artist)
            + F.cross_entropy(outputs["genre"], genre)
            + F.cross_entropy(outputs["style"], style)
        ) / 3.0
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * embeddings.size(0)

    return running_loss / len(dataloader.dataset)


CSV_FIELDNAMES = [
    "epoch",
    "train_loss",
    "mean_train_macro_f1",
    "mean_val_macro_f1",
    "overfit_artist",
    "overfit_genre",
    "overfit_style",
]


def write_metrics_files(
    history: list[dict[str, object]],
    csv_rows: list[dict[str, object]],
    test_metrics: dict[str, dict[str, float]] | None,
    run_summary: dict[str, object],
    output_dir: Path,
) -> None:
    """Persist the full metrics JSON + CSV summary.

    Called after every epoch (with test_metrics=None) so that all per-epoch data is
    saved to disk incrementally and survives an interruption or early stop.
    """
    payload: dict[str, object] = {"run_summary": run_summary, "history": history}
    if test_metrics is not None:
        payload["test_metrics"] = test_metrics
    (output_dir / "clip_classifier_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    import csv

    with open(output_dir / "clip_classifier_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)


def save_plots(csv_rows: list[dict[str, object]], output_dir: Path) -> None:
    """Write loss + mean-macro-F1 curves. No-op if matplotlib is unavailable."""
    if not MATPLOTLIB_AVAILABLE or not csv_rows:
        return
    try:
        epochs = [r["epoch"] for r in csv_rows]
        train_f1 = [r["mean_train_macro_f1"] for r in csv_rows]
        val_f1 = [r["mean_val_macro_f1"] for r in csv_rows]
        losses = [r["train_loss"] for r in csv_rows]

        plt.figure(figsize=(8, 4))
        plt.plot(epochs, losses, marker="o")
        plt.title("Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / "train_loss.png")
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(epochs, train_f1, marker="o", label="train mean macro F1")
        plt.plot(epochs, val_f1, marker="o", label="val mean macro F1")
        plt.title("Mean Macro F1 (train vs val)")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Macro F1")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / "mean_macro_f1.png")
        plt.close()
    except Exception:
        pass


def build_run_summary(
    args: argparse.Namespace,
    epochs_trained: int,
    best_head_f1: dict[str, float],
    best_head_epoch: dict[str, int],
    stopped_early: bool,
) -> dict[str, object]:
    return {
        "epochs_requested": args.epochs,
        "epochs_trained": epochs_trained,
        "early_stop_metric": args.early_stop_metric,
        "best_epoch_per_head": dict(best_head_epoch),
        "best_val_macro_f1_per_head": {head: float(f1) for head, f1 in best_head_f1.items()},
        "mean_best_val_macro_f1": float(np.mean(list(best_head_f1.values()))),
        "patience": args.patience,
        "min_delta": args.min_delta,
        "stopped_early": stopped_early,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_frame = load_manifest(args.split_dir, "train", args.max_train_samples)
    val_frame = load_manifest(args.split_dir, "val", args.max_eval_samples)
    test_frame = load_manifest(args.split_dir, TEST_MANIFEST_NAME, args.max_eval_samples)
    # Build vocab from all frames to ensure coverage even when --max-train-samples limits training
    all_frames = pd.concat([train_frame, val_frame, test_frame], ignore_index=True)
    vocab = LabelVocab.from_frame(all_frames)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHeadClipClassifier(
        model_name=args.model_name,
        vocab=vocab,
        head_type=args.head_type,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    processor = CLIPProcessor.from_pretrained(args.model_name)
    hf_dataset = load_wikiart_dataset()

    # Precompute + cache frozen-CLIP embeddings once per split (the expensive step).
    # Every epoch then trains the tiny heads on these cached vectors -- ~seconds/epoch.
    print("Preparing embeddings (cached after first run)...", flush=True)
    train_emb = get_embeddings(model, processor, hf_dataset, train_frame, "train", args, device)
    val_emb = get_embeddings(model, processor, hf_dataset, val_frame, "val", args, device)
    test_emb = get_embeddings(model, processor, hf_dataset, test_frame, "test", args, device)

    train_loader = make_embedding_loader(train_emb, train_frame, vocab, args.batch_size, True)
    # Eval loader over the training split (no shuffle) to monitor train metrics.
    eval_train_loader = make_embedding_loader(train_emb, train_frame, vocab, args.batch_size, False)
    val_loader = make_embedding_loader(val_emb, val_frame, vocab, args.batch_size, False)
    test_loader = make_embedding_loader(test_emb, test_frame, vocab, args.batch_size, False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )

    history: list[dict[str, object]] = []
    # CSV-friendly summary rows per epoch
    csv_rows: list[dict[str, object]] = []
    # Frozen encoder => the three heads are independent, so we checkpoint each head at
    # its own best-val epoch (genre/style peak early; artist keeps improving later).
    heads = ("artist", "genre", "style")
    best_head_f1 = {head: -1.0 for head in heads}
    best_head_state: dict[str, dict | None] = {head: None for head in heads}
    best_head_epoch = {head: 0 for head in heads}
    best_stop_value = -1.0
    epochs_no_improve = 0
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Evaluate on training split to monitor convergence / overfitting
        train_metrics = evaluate(model, eval_train_loader, device)

        # Evaluate on validation split
        val_metrics = evaluate(model, val_loader, device)

        mean_train_macro_f1 = float(
            np.mean([train_metrics[label]["macro_f1"] for label in ("artist", "genre", "style")])
        )
        mean_val_macro_f1 = float(
            np.mean([val_metrics[label]["macro_f1"] for label in ("artist", "genre", "style")])
        )

        # Simple overfitting heuristic: train macro F1 much higher than val by threshold
        overfit_flags: dict[str, bool] = {}
        OVERFIT_THRESHOLD = 0.10
        for label in ("artist", "genre", "style"):
            overfit_flags[label] = (train_metrics[label]["macro_f1"] - val_metrics[label]["macro_f1"]) > OVERFIT_THRESHOLD

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "mean_train_macro_f1": mean_train_macro_f1,
            "mean_val_macro_f1": mean_val_macro_f1,
            "overfit_flags": overfit_flags,
        }

        history.append(epoch_record)
        print(json.dumps(epoch_record, indent=2))

        # Append CSV-friendly summary
        csv_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "mean_train_macro_f1": mean_train_macro_f1,
                "mean_val_macro_f1": mean_val_macro_f1,
                "overfit_artist": overfit_flags["artist"],
                "overfit_genre": overfit_flags["genre"],
                "overfit_style": overfit_flags["style"],
            }
        )

        # Snapshot each head independently at its own best-val macro-F1 epoch.
        for head in heads:
            head_f1 = val_metrics[head]["macro_f1"]
            if head_f1 > best_head_f1[head]:
                best_head_f1[head] = head_f1
                best_head_epoch[head] = epoch
                head_module = getattr(model, f"{head}_head")
                best_head_state[head] = {
                    key: value.detach().cpu().clone()
                    for key, value in head_module.state_dict().items()
                }

        # Early stopping watches one metric (default: the artist head, the slow learner).
        stop_value = (
            mean_val_macro_f1
            if args.early_stop_metric == "mean"
            else val_metrics[args.early_stop_metric]["macro_f1"]
        )
        if stop_value > best_stop_value + args.min_delta:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        best_stop_value = max(best_stop_value, stop_value)

        run_summary = build_run_summary(args, epoch, best_head_f1, best_head_epoch, False)
        # Persist metrics + plots every epoch so nothing is lost on interruption.
        write_metrics_files(history, csv_rows, None, run_summary, OUTPUT_DIR)
        save_plots(csv_rows, OUTPUT_DIR)

        if args.patience > 0 and epochs_no_improve >= args.patience:
            stopped_early = True
            print(
                f"Early stopping at epoch {epoch}: val '{args.early_stop_metric}' macro-F1 "
                f"has not improved by >{args.min_delta} for {args.patience} epoch(s). "
                f"Best-per-head epochs: {best_head_epoch}."
            )
            break

    # Assemble the final model from each head's own best-val epoch.
    for head in heads:
        if best_head_state[head] is not None:
            getattr(model, f"{head}_head").load_state_dict(best_head_state[head])

    test_metrics = evaluate(model, test_loader, device)

    # Save model checkpoint and vocab (each head restored to its own best-val epoch).
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

    run_summary = build_run_summary(
        args, len(history), best_head_f1, best_head_epoch, stopped_early
    )
    # Final write including the held-out test metrics.
    write_metrics_files(history, csv_rows, test_metrics, run_summary, OUTPUT_DIR)
    save_plots(csv_rows, OUTPUT_DIR)

    print(json.dumps({"run_summary": run_summary, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
