from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset


DATASET_NAME = "huggan/wikiart"
DATASET_SPLIT = "train"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "dataset_analysis"
CACHE_ROOT = (
    Path(os.environ["USERPROFILE"])
    / ".cache"
    / "huggingface"
    / "datasets"
    / "huggan___wikiart"
    / "default"
)


def load_local_cached_dataset() -> Dataset | None:
    if not CACHE_ROOT.exists():
        return None

    candidates = []
    for dataset_info in CACHE_ROOT.rglob("dataset_info.json"):
        parent = dataset_info.parent
        if list(parent.glob("wikiart-train-*.arrow")):
            candidates.append(parent)

    if not candidates:
        return None

    latest_revision = max(candidates, key=lambda path: path.stat().st_mtime)
    arrow_files = sorted(latest_revision.glob("wikiart-train-*.arrow"))

    shards = [Dataset.from_file(str(path)) for path in arrow_files]
    return concatenate_datasets(shards)


def load_wikiart_dataset() -> Dataset:
    cached_dataset = load_local_cached_dataset()
    if cached_dataset is not None:
        return cached_dataset

    return load_dataset(DATASET_NAME, split=DATASET_SPLIT)


def build_distribution_frame(
    counter: Counter[str], all_labels: list[str], label_type: str
) -> pd.DataFrame:
    rows = []
    total = sum(counter.values())

    for label in all_labels:
        count = counter.get(label, 0)
        percent = (count / total * 100) if total else 0.0
        rows.append(
            {
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


def summarize_bias(frame: pd.DataFrame, top_n: int = 10) -> dict[str, object]:
    total = int(frame["count"].sum())
    top_slice = frame.head(top_n)
    nonzero = frame[frame["count"] > 0]

    return {
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
    destination: Path, dataset_rows: int, summaries: dict[str, dict[str, object]]
) -> None:
    lines = [
        "# WikiArt Dataset Label Audit",
        "",
        f"- Dataset: `{DATASET_NAME}`",
        f"- Split: `{DATASET_SPLIT}`",
        f"- Total rows processed: `{dataset_rows}`",
        "",
        "## Bias Summary",
        "",
    ]

    for label_type, summary in summaries.items():
        lines.extend(
            [
                f"### {label_type.title()}",
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

    dataset = load_wikiart_dataset().remove_columns("image")

    features = dataset.features
    artist_labels = list(features["artist"].names)
    genre_labels = list(features["genre"].names)
    style_labels = list(features["style"].names)

    artist_indices = dataset["artist"]
    genre_indices = dataset["genre"]
    style_indices = dataset["style"]

    artist_counts = Counter(artist_labels[index] for index in artist_indices)
    genre_counts = Counter(genre_labels[index] for index in genre_indices)
    style_counts = Counter(style_labels[index] for index in style_indices)
    total_rows = dataset.num_rows

    distributions = {
        "artist": build_distribution_frame(artist_counts, artist_labels, "artist"),
        "genre": build_distribution_frame(genre_counts, genre_labels, "genre"),
        "style": build_distribution_frame(style_counts, style_labels, "style"),
    }

    summaries = {
        label_type: summarize_bias(frame)
        for label_type, frame in distributions.items()
    }

    for label_type, frame in distributions.items():
        frame.to_csv(OUTPUT_DIR / f"{label_type}_distribution.csv", index=False)

    combined = pd.concat(distributions.values(), ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "all_label_distributions.csv", index=False)

    labels_only = {
        "artist": artist_labels,
        "genre": genre_labels,
        "style": style_labels,
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
                "total_rows": total_rows,
                "summaries": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown_report(OUTPUT_DIR / "bias_report.md", total_rows, summaries)

    print(f"Processed {total_rows} rows from {DATASET_NAME}.")
    for label_type, frame in distributions.items():
        print(f"\nTop {label_type} labels by count:")
        print(frame.head(10).to_string(index=False))
        print(f"\nSaved: {OUTPUT_DIR / f'{label_type}_distribution.csv'}")


if __name__ == "__main__":
    main()
