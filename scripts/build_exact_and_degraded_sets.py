"""Build the "exact artwork" and "degraded exact artwork" evaluation sets.

Unlike ``known_artworks_test`` (held-out images from *seen* classes -> generalization),
the exact-artwork set is a random sample of images the model WAS trained on. It gives a
memorization / upper-bound reference and, paired with its degraded version, answers the
"museum photo of an artwork we've catalogued" scenario.

Outputs (under outputs/phase0/):
  - exact_artworks_test_manifest.csv         : the sampled training rows (matched to test size)
  - degraded_exact_artworks_manifest.csv     : all corruption recipes applied to that sample
  - degraded_exact_artworks_preview/         : a few materialized images for visual QA

The full degraded set is NOT materialized by default -- eval_retrieval.py applies the
(deterministic) corruptions on the fly. Pass --materialize full to render every image.

Examples:
    python scripts/build_exact_and_degraded_sets.py
    python scripts/build_exact_and_degraded_sets.py --materialize full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import datasets (pyarrow) first to avoid the Windows DLL load-order crash seen elsewhere.
import datasets  # noqa: F401  (import-order side effect; used via data.wikiart_utils)

import numpy as np
import pandas as pd

from data.image_corruptions import CORRUPTION_RECIPES, apply_corruption, ensure_parent
from data.wikiart_utils import load_wikiart_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE0_DIR = REPO_ROOT / "outputs" / "phase0"

DEFAULT_RECIPES = [
    "gaussian_blur",
    "strong_gaussian_blur",
    "small_center_crop",
    "medium_center_crop",
    "warm_shift",
    "desaturated",
    "tilt_left",
    "tilt_right",
    "perspective_left",
    "perspective_right",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build exact + degraded exact-artwork eval sets.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Rows to sample from train. Default: match the known_artworks_test set size.",
    )
    parser.add_argument(
        "--recipes",
        nargs="*",
        default=None,
        help="Override the corruption recipe list. Default: all 10 museum recipes.",
    )
    parser.add_argument(
        "--materialize",
        choices=("none", "preview", "full", "sampled"),
        default="preview",
        help="none: manifests only. preview (default): render a few images per recipe for QA; "
        "the full set is streamed on the fly at eval time. "
        "sampled: render one balanced recipe per artwork (~val/test-sized set). "
        "full: render every degraded image to disk (~10x sample-size JPEGs).",
    )
    parser.add_argument(
        "--sampled-size",
        type=int,
        default=None,
        help="Size of the --materialize sampled set. Default: match known_artworks_test.",
    )
    parser.add_argument(
        "--preview-artworks",
        type=int,
        default=4,
        help="How many base artworks to render across all recipes when --materialize preview.",
    )
    return parser.parse_args()


def build_degraded_manifest(base: pd.DataFrame, recipes: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in base.iterrows():
        record = row.to_dict()
        for corruption_name in recipes:
            degraded = dict(record)
            degraded["source_split"] = "exact_artworks"
            degraded["corruption"] = corruption_name
            degraded["degraded_image_relpath"] = (
                f"degraded_exact_artworks/{corruption_name}/row_{int(row['row_id']):06d}.jpg"
            )
            rows.append(degraded)
    return pd.DataFrame(rows)


def materialize(manifest: pd.DataFrame, output_root: Path) -> int:
    dataset = load_wikiart_dataset()
    rendered = 0
    for row in manifest.to_dict(orient="records"):
        image = dataset[int(row["row_id"])]["image"].convert("RGB")
        corrupted = apply_corruption(image, str(row["corruption"]))
        destination = output_root / str(row["degraded_image_relpath"])
        ensure_parent(destination)
        corrupted.save(destination, format="JPEG", quality=95)
        rendered += 1
        if rendered % 2000 == 0:
            print(f"    rendered {rendered}/{len(manifest)}", flush=True)
    return rendered


def main() -> None:
    args = parse_args()
    PHASE0_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(PHASE0_DIR / "train_manifest.csv")
    test = pd.read_csv(PHASE0_DIR / "known_artworks_test_manifest.csv")
    sample_size = args.sample_size or len(test)
    sample_size = min(sample_size, len(train))

    exact = train.sample(n=sample_size, random_state=args.seed).reset_index(drop=True)
    exact["source_split"] = "exact_artworks"
    exact_path = PHASE0_DIR / "exact_artworks_test_manifest.csv"
    exact.to_csv(exact_path, index=False)

    recipes = args.recipes or DEFAULT_RECIPES
    degraded = build_degraded_manifest(exact, recipes)
    degraded_path = PHASE0_DIR / "degraded_exact_artworks_manifest.csv"
    degraded.to_csv(degraded_path, index=False)

    rendered = 0
    preview_root = PHASE0_DIR
    if args.materialize == "full":
        print(f"Materializing all {len(degraded)} degraded images...", flush=True)
        rendered = materialize(degraded, PHASE0_DIR)
    elif args.materialize == "preview":
        preview_ids = exact["row_id"].head(args.preview_artworks).tolist()
        preview = degraded[degraded["row_id"].isin(preview_ids)].copy()
        # Redirect preview renders into a clearly separate folder.
        preview["degraded_image_relpath"] = preview["degraded_image_relpath"].str.replace(
            "degraded_exact_artworks/", "degraded_exact_artworks_preview/", regex=False
        )
        print(
            f"Rendering preview: {args.preview_artworks} artworks x {len(recipes)} recipes...",
            flush=True,
        )
        rendered = materialize(preview, PHASE0_DIR)

    summary = {
        "seed": args.seed,
        "sample_size": int(sample_size),
        "matched_to": "known_artworks_test",
        "recipes": recipes,
        "exact_manifest": str(exact_path.relative_to(REPO_ROOT)),
        "exact_rows": int(len(exact)),
        "degraded_manifest": str(degraded_path.relative_to(REPO_ROOT)),
        "degraded_rows": int(len(degraded)),
        "materialize_mode": args.materialize,
        "images_rendered": int(rendered),
        "unique_artists_in_sample": int(exact["artist"].nunique()),
        "unique_styles_in_sample": int(exact["style"].nunique()),
        "unique_genres_in_sample": int(exact["genre"].nunique()),
    }
    (PHASE0_DIR / "exact_and_degraded_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
