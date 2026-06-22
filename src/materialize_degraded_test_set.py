from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from image_corruptions import apply_corruption, ensure_parent
from wikiart_utils import load_wikiart_dataset


DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "outputs" / "phase0" / "degraded_known_artworks_manifest.csv"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "phase0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize degraded known-artwork test images from the Phase 0 manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally render only the first N degraded images for a quick smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    if args.limit is not None:
        manifest = manifest.head(args.limit).copy()

    dataset = load_wikiart_dataset()

    for row in manifest.to_dict(orient="records"):
        image = dataset[int(row["row_id"])]["image"].convert("RGB")
        corrupted = apply_corruption(image, str(row["corruption"]))
        destination = args.output_root / str(row["degraded_image_relpath"])
        ensure_parent(destination)
        corrupted.save(destination, format="JPEG", quality=95)

    print(f"Rendered {len(manifest)} degraded images into {args.output_root}.")


if __name__ == "__main__":
    main()
