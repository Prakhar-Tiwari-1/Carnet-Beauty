"""Guard for every file written to app_data/ (the only data the deployed app sees)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from sillage.config import load_config

FORBIDDEN_COLUMNS = {
    "author_id", "review_text", "review_title", "skin_tone", "eye_color", "hair_color",
}


class ExportError(ValueError):
    pass


def check_export(df: pd.DataFrame, count_columns: tuple[str, ...] = ()) -> None:
    """Refuse identifying columns and any displayed aggregate built on too few reviews."""
    bad = FORBIDDEN_COLUMNS & {c.lower() for c in df.columns}
    if bad:
        raise ExportError(f"Forbidden columns in export: {sorted(bad)}")
    min_n = load_config()["min_group_size"]
    for c in count_columns:
        small = df[c] < min_n
        if small.any():
            raise ExportError(f"{int(small.sum())} rows with {c} < {min_n}; suppress them first")


def write_export(df: pd.DataFrame, path: Path, count_columns: tuple[str, ...] = ()) -> None:
    check_export(df, count_columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
