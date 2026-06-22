from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset


DATASET_NAME = "huggan/wikiart"
DATASET_SPLIT = "train"
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

    candidates: list[Path] = []
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


def dataset_to_metadata_frame(dataset: Dataset) -> pd.DataFrame:
    features = dataset.features
    artist_labels = list(features["artist"].names)
    genre_labels = list(features["genre"].names)
    style_labels = list(features["style"].names)

    frame = pd.DataFrame(
        {
            "row_id": list(range(dataset.num_rows)),
            "artist_id": dataset["artist"],
            "genre_id": dataset["genre"],
            "style_id": dataset["style"],
        }
    )
    frame["artist"] = frame["artist_id"].map(lambda idx: artist_labels[idx])
    frame["genre"] = frame["genre_id"].map(lambda idx: genre_labels[idx])
    frame["style"] = frame["style_id"].map(lambda idx: style_labels[idx])
    return frame
