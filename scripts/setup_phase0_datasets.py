from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from data.split_utils import build_stratify_key, split_frame, summarize_split
from data.wikiart_utils import dataset_to_metadata_frame, load_wikiart_dataset


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "phase0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set up Phase 0 WikiArt datasets with Unknown Artist removed."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--min-combo-count", type=int, default=5)
    parser.add_argument(
        "--keep-unknown-artist",
        action="store_true",
        help="Keep `Unknown Artist` rows. Default behavior removes them.",
    )
    parser.add_argument(
        "--degradation-recipes",
        nargs="*",
        default=None,
        help="Optional override for the degraded test-set recipe list.",
    )
    return parser.parse_args()


def summarize_filtered_dataset(frame: pd.DataFrame) -> dict[str, object]:
    artist_counts = frame["artist"].value_counts()
    style_counts = frame["style"].value_counts()
    genre_counts = frame["genre"].value_counts()
    return {
        "rows": int(len(frame)),
        "unique_artists": int(frame["artist"].nunique()),
        "unique_styles": int(frame["style"].nunique()),
        "unique_genres": int(frame["genre"].nunique()),
        "unknown_artist_rows": int((frame["artist"] == "Unknown Artist").sum()),
        "unknown_genre_rows": int((frame["genre"] == "Unknown Genre").sum()),
        "largest_artist_label": str(artist_counts.index[0]),
        "largest_artist_count": int(artist_counts.iloc[0]),
        "largest_style_label": str(style_counts.index[0]),
        "largest_style_count": int(style_counts.iloc[0]),
        "largest_genre_label": str(genre_counts.index[0]),
        "largest_genre_count": int(genre_counts.iloc[0]),
    }


def coverage_report(
    full_frame: pd.DataFrame,
    split_frame: pd.DataFrame,
    label_column: str,
) -> dict[str, object]:
    full_labels = set(full_frame[label_column])
    split_labels = set(split_frame[label_column])
    missing = sorted(full_labels - split_labels)
    return {
        "present_count": int(len(split_labels)),
        "full_count": int(len(full_labels)),
        "missing_labels": missing,
    }


def build_degraded_manifest(
    clean_test: pd.DataFrame,
    degradation_recipes: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in clean_test.iterrows():
        base = row.to_dict()
        for corruption_name in degradation_recipes:
            degraded = dict(base)
            degraded["source_split"] = "known_artworks_test"
            degraded["corruption"] = corruption_name
            degraded["degraded_image_relpath"] = (
                f"degraded_known_artworks/{corruption_name}/row_{int(row['row_id']):06d}.jpg"
            )
            rows.append(degraded)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_wikiart_dataset()
    frame = dataset_to_metadata_frame(dataset)
    removed_unknown_artist_rows = int((frame["artist"] == "Unknown Artist").sum())

    if not args.keep_unknown_artist:
        frame = frame[frame["artist"] != "Unknown Artist"].reset_index(drop=True)

    frame["stratify_key"] = build_stratify_key(frame, args.min_combo_count)
    train, val, test = split_frame(
        frame=frame,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    recipes = args.degradation_recipes or [
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
    degraded_test = build_degraded_manifest(test, recipes)

    train.to_csv(OUTPUT_DIR / "train_manifest.csv", index=False)
    val.to_csv(OUTPUT_DIR / "val_manifest.csv", index=False)
    test.to_csv(OUTPUT_DIR / "known_artworks_test_manifest.csv", index=False)
    degraded_test.to_csv(OUTPUT_DIR / "degraded_known_artworks_manifest.csv", index=False)

    metadata = {
        "seed": args.seed,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "min_combo_count": args.min_combo_count,
        "unknown_artist_removed": not args.keep_unknown_artist,
        "removed_unknown_artist_rows": removed_unknown_artist_rows if not args.keep_unknown_artist else 0,
        "filtered_dataset_summary": summarize_filtered_dataset(frame),
        "split_summary": {
            "train": summarize_split(train),
            "val": summarize_split(val),
            "known_artworks_test": summarize_split(test),
        },
        "label_coverage": {
            "val": {
                "artist": coverage_report(frame, val, "artist"),
                "style": coverage_report(frame, val, "style"),
                "genre": coverage_report(frame, val, "genre"),
            },
            "known_artworks_test": {
                "artist": coverage_report(frame, test, "artist"),
                "style": coverage_report(frame, test, "style"),
                "genre": coverage_report(frame, test, "genre"),
            },
        },
        "degraded_known_artworks": {
            "source_rows": int(len(test)),
            "recipes": recipes,
            "total_rows": int(len(degraded_test)),
        },
    }
    (OUTPUT_DIR / "phase0_dataset_summary.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
