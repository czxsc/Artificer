from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from data.wikiart_utils import (
    DATASET_NAME,
    DATASET_SPLIT,
    dataset_to_metadata_frame,
    load_wikiart_dataset,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "dataset_analysis"
UNKNOWN_ARTIST_LABEL = "Unknown Artist"


def build_distribution_frame(
    counter: Counter[str],
    all_labels: list[str],
    label_type: str,
    dataset_variant: str,
) -> pd.DataFrame:
    rows = []
    total = sum(counter.values())

    for label in all_labels:
        count = counter.get(label, 0)
        percent = (count / total * 100) if total else 0.0
        rows.append(
            {
                "dataset_variant": dataset_variant,
                "label_type": label_type,
                "label": label,
                "count": count,
                "percent": round(percent, 4),
            }
        )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["count", "label"], ascending=[False, True]).reset_index(
        drop=True
    )
    return frame


def summarize_bias(
    frame: pd.DataFrame, total_rows: int, top_n: int = 10
) -> dict[str, object]:
    total = int(frame["count"].sum())
    top_slice = frame.head(top_n)
    nonzero = frame[frame["count"] > 0]

    return {
        "total_rows": total_rows,
        "num_labels": int(len(frame)),
        "num_nonzero_labels": int(len(nonzero)),
        "largest_label": str(frame.iloc[0]["label"]),
        "largest_count": int(frame.iloc[0]["count"]),
        "largest_percent": float(frame.iloc[0]["percent"]),
        "smallest_nonzero_label": (
            str(nonzero.iloc[-1]["label"]) if not nonzero.empty else None
        ),
        "smallest_nonzero_count": (
            int(nonzero.iloc[-1]["count"]) if not nonzero.empty else None
        ),
        "top_10_share_percent": (
            round(float(top_slice["count"].sum()) / total * 100, 4) if total else 0.0
        ),
        "single_sample_labels": int((frame["count"] == 1).sum()),
        "labels_under_10_samples": int((frame["count"] < 10).sum()),
        "labels_under_50_samples": int((frame["count"] < 50).sum()),
    }


def write_markdown_report(
    destination: Path, summaries: dict[str, dict[str, dict[str, object]]]
) -> None:
    lines = [
        "# WikiArt Dataset Label Audit",
        "",
        f"- Dataset: `{DATASET_NAME}`",
        f"- Split: `{DATASET_SPLIT}`",
        "",
        "## Bias Summary",
        "",
    ]

    for dataset_variant, variant_summaries in summaries.items():
        lines.extend(
            [
                f"### {dataset_variant}",
                "",
                f"- Total rows processed: `{variant_summaries['artist']['total_rows']}`",
                "",
            ]
        )

        for label_type, summary in variant_summaries.items():
            lines.extend(
                [
                    f"#### {label_type.title()}",
                    "",
                    f"- Total labels: `{summary['num_labels']}`",
                    f"- Labels with at least one sample: `{summary['num_nonzero_labels']}`",
                    f"- Largest class: `{summary['largest_label']}` with `{summary['largest_count']}` works (`{summary['largest_percent']}%`)",
                    f"- Smallest non-zero class: `{summary['smallest_nonzero_label']}` with `{summary['smallest_nonzero_count']}` works",
                    f"- Top 10 classes share: `{summary['top_10_share_percent']}%`",
                    f"- Labels with 1 sample: `{summary['single_sample_labels']}`",
                    f"- Labels with fewer than 10 samples: `{summary['labels_under_10_samples']}`",
                    f"- Labels with fewer than 50 samples: `{summary['labels_under_50_samples']}`",
                    "",
                ]
            )

    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_wikiart_dataset()
    metadata = dataset_to_metadata_frame(dataset)

    features = dataset.features
    labels_by_type = {
        "artist": list(features["artist"].names),
        "genre": list(features["genre"].names),
        "style": list(features["style"].names),
    }

    dataset_variants = {
        "full_dataset": metadata,
        "known_artist_only": metadata[metadata["artist"] != UNKNOWN_ARTIST_LABEL].copy(),
    }

    all_distribution_frames: list[pd.DataFrame] = []
    all_summaries: dict[str, dict[str, dict[str, object]]] = {}

    for dataset_variant, variant_frame in dataset_variants.items():
        distributions = {}
        summaries = {}

        for label_type, all_labels in labels_by_type.items():
            counter = Counter(variant_frame[label_type])
            distribution = build_distribution_frame(
                counter=counter,
                all_labels=all_labels,
                label_type=label_type,
                dataset_variant=dataset_variant,
            )
            distributions[label_type] = distribution
            summaries[label_type] = summarize_bias(
                distribution, total_rows=len(variant_frame)
            )
            distribution.to_csv(
                OUTPUT_DIR / f"{dataset_variant}_{label_type}_distribution.csv",
                index=False,
            )

        all_distribution_frames.extend(distributions.values())
        all_summaries[dataset_variant] = summaries

    combined = pd.concat(all_distribution_frames, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "all_label_distributions.csv", index=False)

    labels_only = {
        "artist": labels_by_type["artist"],
        "genre": labels_by_type["genre"],
        "style": labels_by_type["style"],
    }
    (OUTPUT_DIR / "label_vocabularies.json").write_text(
        json.dumps(labels_only, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "dataset_name": DATASET_NAME,
                "split": DATASET_SPLIT,
                "variant_row_counts": {
                    dataset_variant: int(len(variant_frame))
                    for dataset_variant, variant_frame in dataset_variants.items()
                },
                "summaries": all_summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown_report(OUTPUT_DIR / "bias_report.md", all_summaries)

    print(f"Processed {len(metadata)} rows from {DATASET_NAME}.")
    print(
        f"Known-artist subset rows after filtering '{UNKNOWN_ARTIST_LABEL}': "
        f"{len(dataset_variants['known_artist_only'])}"
    )
    for dataset_variant in dataset_variants:
        print(f"\nTop labels for {dataset_variant}:")
        for label_type in ("artist", "genre", "style"):
            output_path = OUTPUT_DIR / f"{dataset_variant}_{label_type}_distribution.csv"
            frame = combined[
                (combined["dataset_variant"] == dataset_variant)
                & (combined["label_type"] == label_type)
            ]
            print(f"\n{label_type.title()}:")
            print(frame.head(10).to_string(index=False))
            print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
