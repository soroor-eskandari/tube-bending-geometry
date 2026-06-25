from pathlib import Path

import pandas as pd


def parquet_path(path) -> Path:
    path = Path(path)
    return path.with_suffix(".parquet")


def csv_path(path) -> Path:
    path = Path(path)
    return path.with_suffix(".csv")


def existing_table_path(path) -> Path:
    path = Path(path)
    candidates = []

    if path.suffix in {".parquet", ".csv"}:
        candidates.append(path)
        candidates.append(parquet_path(path))
        candidates.append(csv_path(path))
    else:
        candidates.append(path.with_suffix(".parquet"))
        candidates.append(path.with_suffix(".csv"))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    return candidates[0]


def read_table(path) -> pd.DataFrame:
    resolved_path = existing_table_path(path)
    if resolved_path.suffix == ".parquet":
        return pd.read_parquet(resolved_path)
    if resolved_path.suffix == ".csv":
        return pd.read_csv(resolved_path)
    raise ValueError(f"Unsupported table format: {resolved_path}")


def write_table(df: pd.DataFrame, path, index: bool = False) -> Path:
    output_path = parquet_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=index)
    return output_path
