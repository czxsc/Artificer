"""Evaluate a cached retrieval index and append results to a cross-encoder comparison.

Measures, for a set of query images:
  - Exact retrieval  : is the query's own artwork (row_id) the top-1 / within top-k?
                       (only scored for queries whose row_id exists in the database)
  - Label agreement  : does the cosine-weighted vote over the top-k neighbors predict
                       the correct artist / genre / style?
  - Latency          : mean encode + search time per query.

Queries can be corrupted on the fly (--corruption gaussian_blur, tilt_left, ...) to
test robustness, reusing the Phase 0 corruption recipes.

Examples:
    # Self-retrieval sanity check (exclude the identical vector):
    python scripts/eval_retrieval.py --encoder clip --exclude-self
    # Degraded-known robustness (find the ORIGINAL from a blurred query):
    python scripts/eval_retrieval.py --encoder clip --corruption gaussian_blur
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import datasets (pyarrow) before torch to avoid a Windows DLL load-order crash.
import datasets  # noqa: F401  (import-order side effect; used via data.wikiart_utils)

import numpy as np
import pandas as pd
import torch

from data.image_corruptions import CORRUPTION_RECIPES, apply_corruption
from data.wikiart_utils import load_wikiart_dataset
from ml.models.retrieval import ENCODERS, LABEL_COLUMNS, RetrievalIndex, get_encoder

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERY_MANIFEST = REPO_ROOT / "outputs" / "phase0" / "known_artworks_test_manifest.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "retrieval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a retrieval index over a query set.")
    parser.add_argument("--encoder", choices=sorted(ENCODERS), required=True)
    parser.add_argument("--index-dir", type=Path, default=None, help="Defaults to outputs/retrieval/<encoder>.")
    parser.add_argument("--query-manifest", type=Path, default=DEFAULT_QUERY_MANIFEST)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--corruption",
        choices=sorted(CORRUPTION_RECIPES),
        default=None,
        help="Apply this corruption to every query image before encoding.",
    )
    parser.add_argument(
        "--exclude-self",
        action="store_true",
        help="Drop the query's own row_id from results (use for clean self-retrieval).",
    )
    parser.add_argument("--max-queries", type=int, default=None)
    return parser.parse_args()


def load_query_images(hf_dataset, manifest: pd.DataFrame, corruption: str | None) -> list:
    images = []
    for row_id in manifest["row_id"]:
        image = hf_dataset[int(row_id)]["image"].convert("RGB")
        if corruption is not None:
            image = apply_corruption(image, corruption)
        images.append(image)
    return images


def encode_all(encoder, images: list, batch_size: int) -> tuple[np.ndarray, float]:
    vectors: list[np.ndarray] = []
    started = time.perf_counter()
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        vectors.append(encoder.encode_images(batch).cpu().numpy().astype("float32"))
    elapsed = time.perf_counter() - started
    return np.concatenate(vectors, axis=0), elapsed


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index_dir = args.index_dir or (args.out_root / args.encoder)

    index = RetrievalIndex.load(index_dir)
    db_row_ids = set(index.metadata["row_id"].tolist())

    manifest = pd.read_csv(args.query_manifest)
    if args.max_queries is not None:
        manifest = manifest.head(args.max_queries).copy()
    manifest = manifest.reset_index(drop=True)

    encoder = get_encoder(args.encoder, device=device)
    hf_dataset = load_wikiart_dataset()

    images = load_query_images(hf_dataset, manifest, args.corruption)
    query_emb, encode_seconds = encode_all(encoder, images, args.batch_size)

    # Search a little deeper so exclude-self can drop the identical hit and still return k.
    search_k = args.k + 1 if args.exclude_self else args.k
    started = time.perf_counter()
    scores, indices = index.search(query_emb, search_k)
    search_seconds = time.perf_counter() - started

    retrieved_row_ids = index.metadata["row_id"].to_numpy()[indices]  # [Q, search_k]

    exact_top1_hits: list[bool] = []
    exact_topk_hits: list[bool] = []
    label_top1_hits: dict[str, list[bool]] = {col: [] for col in LABEL_COLUMNS}

    for q, (query_row) in enumerate(manifest.itertuples(index=False)):
        query_row_id = int(getattr(query_row, "row_id"))
        row_ids = retrieved_row_ids[q]
        row_scores = scores[q]

        if args.exclude_self:
            keep = row_ids != query_row_id
            row_ids = row_ids[keep][: args.k]
            neighbor_positions = indices[q][keep][: args.k]
            neighbor_scores = row_scores[keep][: args.k]
        else:
            row_ids = row_ids[: args.k]
            neighbor_positions = indices[q][: args.k]
            neighbor_scores = row_scores[: args.k]

        # Exact retrieval only makes sense when the query artwork is in the database.
        if query_row_id in db_row_ids:
            exact_top1_hits.append(bool(row_ids[0] == query_row_id))
            exact_topk_hits.append(bool(query_row_id in set(row_ids.tolist())))

        aggregated = index.aggregate_labels(
            neighbor_positions[None, :], neighbor_scores[None, :], method="cosine_weighted"
        )[0]
        for col in LABEL_COLUMNS:
            predicted = aggregated[col][0]["label"] if aggregated[col] else None
            label_top1_hits[col].append(predicted == getattr(query_row, col))

    def mean(values: list[bool]) -> float | None:
        return float(np.mean(values)) if values else None

    num_queries = len(manifest)
    metrics = {
        "exact_retrieval": {
            "scored_queries": len(exact_top1_hits),
            "top1_accuracy": mean(exact_top1_hits),
            f"top{args.k}_accuracy": mean(exact_topk_hits),
        },
        "label_top1_accuracy": {col: mean(label_top1_hits[col]) for col in LABEL_COLUMNS},
        "label_top1_accuracy_mean": mean(
            [hit for col in LABEL_COLUMNS for hit in label_top1_hits[col]]
        ),
        "latency_ms_per_query": {
            "encode": 1000.0 * encode_seconds / max(num_queries, 1),
            "search": 1000.0 * search_seconds / max(num_queries, 1),
        },
    }

    config = {
        "encoder": args.encoder,
        "model_id": index.model_id,
        "embed_dim": int(index.embeddings.shape[1]),
        "db_items": int(index.embeddings.shape[0]),
        "query_manifest": args.query_manifest.name,
        "corruption": args.corruption or "none",
        "k": args.k,
        "exclude_self": args.exclude_self,
        "num_queries": num_queries,
        "device": str(device),
    }
    payload = {"config": config, "metrics": metrics}

    out_dir = args.out_root / args.encoder
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{Path(args.query_manifest).stem}__{args.corruption or 'clean'}"
    (out_dir / f"eval_{tag}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _append_comparison_row(args.out_root / "comparison.csv", config, metrics)

    print(json.dumps(payload, indent=2))


def _append_comparison_row(csv_path: Path, config: dict, metrics: dict) -> None:
    """Upsert one flat row into a shared cross-encoder comparison table."""
    row = {
        "encoder": config["encoder"],
        "query_set": config["query_manifest"],
        "corruption": config["corruption"],
        "k": config["k"],
        "exclude_self": config["exclude_self"],
        "exact_top1": metrics["exact_retrieval"]["top1_accuracy"],
        f"exact_top{config['k']}": metrics["exact_retrieval"][f"top{config['k']}_accuracy"],
        "artist_top1": metrics["label_top1_accuracy"]["artist"],
        "genre_top1": metrics["label_top1_accuracy"]["genre"],
        "style_top1": metrics["label_top1_accuracy"]["style"],
        "label_top1_mean": metrics["label_top1_accuracy_mean"],
        "encode_ms": round(metrics["latency_ms_per_query"]["encode"], 2),
        "search_ms": round(metrics["latency_ms_per_query"]["search"], 3),
    }
    key_cols = ["encoder", "query_set", "corruption", "k", "exclude_self"]
    if csv_path.exists():
        table = pd.read_csv(csv_path)
        mask = np.ones(len(table), dtype=bool)
        for col in key_cols:
            mask &= table[col].astype(str) == str(row[col])
        table = table[~mask]
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    else:
        table = pd.DataFrame([row])
    table.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
