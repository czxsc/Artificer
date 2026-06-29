from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from data.split_utils import build_stratify_key, split_frame, summarize_split
from data.wikiart_utils import dataset_to_metadata_frame, load_wikiart_dataset


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "phase1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare reproducible stratified WikiArt manifests for Phase 1."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument(
        "--min-combo-count",
        type=int,
        default=5,
        help="Minimum frequency for a full artist/genre/style combination before falling back.",
    )
    parser.add_argument(
        "--drop-unknown-artist",
        action="store_true",
        help="Exclude the dominant 'Unknown Artist' class for cleaner artist experiments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_wikiart_dataset()
    frame = dataset_to_metadata_frame(dataset)

    if args.drop_unknown_artist:
        frame = frame[frame["artist"] != "Unknown Artist"].reset_index(drop=True)

    frame["stratify_key"] = build_stratify_key(frame, args.min_combo_count)
    train, val, test = split_frame(
        frame=frame,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    train.to_csv(OUTPUT_DIR / "train_manifest.csv", index=False)
    val.to_csv(OUTPUT_DIR / "val_manifest.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test_manifest.csv", index=False)

    metadata = {
        "seed": args.seed,
        "dataset_rows": int(len(frame)),
        "test_size": args.test_size,
        "val_size": args.val_size,
        "min_combo_count": args.min_combo_count,
        "drop_unknown_artist": args.drop_unknown_artist,
        "split_summary": {
            "train": summarize_split(train),
            "val": summarize_split(val),
            "test": summarize_split(test),
        },
    }
    (OUTPUT_DIR / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
