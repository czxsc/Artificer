from __future__ import annotations

import numpy as np
import pandas as pd


def build_stratify_key(frame: pd.DataFrame, min_combo_count: int) -> pd.Series:
    combo = frame["artist"] + "||" + frame["genre"] + "||" + frame["style"]
    combo_counts = combo.value_counts()

    style_artist = frame["style"] + "||" + frame["artist"]
    style_artist_counts = style_artist.value_counts()
    style_counts = frame["style"].value_counts()
    artist_counts = frame["artist"].value_counts()
    genre_counts = frame["genre"].value_counts()

    key = pd.Series(index=frame.index, dtype="object")
    combo_mask = combo.map(combo_counts) >= min_combo_count
    style_artist_mask = style_artist.map(style_artist_counts) >= min_combo_count
    style_mask = frame["style"].map(style_counts) >= min_combo_count
    artist_mask = frame["artist"].map(artist_counts) >= min_combo_count
    genre_mask = frame["genre"].map(genre_counts) >= min_combo_count

    key.loc[combo_mask] = "combo::" + combo.loc[combo_mask]
    key.loc[key.isna() & style_artist_mask] = (
        "style_artist::" + style_artist.loc[key.isna() & style_artist_mask]
    )
    key.loc[key.isna() & style_mask] = "style::" + frame.loc[key.isna() & style_mask, "style"]
    key.loc[key.isna() & artist_mask] = (
        "artist::" + frame.loc[key.isna() & artist_mask, "artist"]
    )
    key.loc[key.isna() & genre_mask] = "genre::" + frame.loc[key.isna() & genre_mask, "genre"]
    return key.fillna("__rare__")


def split_frame(
    frame: pd.DataFrame, seed: int, test_size: float, val_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if test_size + val_size >= 1.0:
        raise ValueError("Validation/test sizes leave no room for a training split.")

    rng = np.random.default_rng(seed)
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, group in frame.groupby("stratify_key", sort=False):
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)
        n_rows = len(indices)

        test_count = int(round(n_rows * test_size))
        val_count = int(round(n_rows * val_size))

        while n_rows - test_count - val_count < 1:
            if test_count >= val_count and test_count > 0:
                test_count -= 1
            elif val_count > 0:
                val_count -= 1
            else:
                break

        test_idx = indices[:test_count]
        val_idx = indices[test_count : test_count + val_count]
        train_idx = indices[test_count + val_count :]

        train_parts.append(frame.loc[train_idx])
        if len(val_idx):
            val_parts.append(frame.loc[val_idx])
        if len(test_idx):
            test_parts.append(frame.loc[test_idx])

    train = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train, val, test


def summarize_split(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(frame)),
        "unique_artists": int(frame["artist"].nunique()),
        "unique_genres": int(frame["genre"].nunique()),
        "unique_styles": int(frame["style"].nunique()),
        "unknown_artist_rows": int((frame["artist"] == "Unknown Artist").sum()),
        "unknown_genre_rows": int((frame["genre"] == "Unknown Genre").sum()),
    }
