"""Build and cache a retrieval database (image embeddings + label metadata).

Encodes every artwork in a manifest with the chosen vision encoder and saves the
embedding matrix + metadata under outputs/retrieval/<encoder>/. Run once per encoder;
eval_retrieval.py then loads the cached index.

Examples:
    python scripts/build_retrieval_index.py --encoder clip
    python scripts/build_retrieval_index.py --encoder dino --db-manifest outputs/phase0/train_manifest.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import datasets (pyarrow) before torch to avoid a Windows DLL load-order crash.
import datasets  # noqa: F401  (import-order side effect; used via data.wikiart_utils)

import pandas as pd
import torch

from data.wikiart_utils import load_wikiart_dataset
from ml.models.retrieval import ENCODERS, RetrievalIndex, get_encoder

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_MANIFEST = REPO_ROOT / "outputs" / "phase0" / "train_manifest.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "retrieval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cached retrieval index for one encoder.")
    parser.add_argument("--encoder", choices=sorted(ENCODERS), required=True)
    parser.add_argument("--db-manifest", type=Path, default=DEFAULT_DB_MANIFEST)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Only index the first N artworks (quick smoke test).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = pd.read_csv(args.db_manifest)
    if args.max_items is not None:
        manifest = manifest.head(args.max_items).copy()

    print(f"Loading encoder '{args.encoder}' on {device}...", flush=True)
    encoder = get_encoder(args.encoder, device=device)
    print(f"  model_id={encoder.model_id} embed_dim={encoder.embed_dim}", flush=True)

    hf_dataset = load_wikiart_dataset()

    print(f"Encoding {len(manifest)} artworks into the index...", flush=True)
    started = time.perf_counter()
    index = RetrievalIndex.build(encoder, hf_dataset, manifest, batch_size=args.batch_size)
    elapsed = time.perf_counter() - started

    out_dir = args.out_root / args.encoder
    index.save(out_dir)
    print(
        f"Saved index to {out_dir} "
        f"({index.embeddings.shape[0]} items x {index.embeddings.shape[1]}-d, "
        f"{elapsed:.1f}s, {index.embeddings.shape[0] / max(elapsed, 1e-6):.1f} img/s).",
        flush=True,
    )


if __name__ == "__main__":
    main()
