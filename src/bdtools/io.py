from pathlib import Path

import pandas as pd

from bdtools.cleaning import clean_dataset, quality_report
from bdtools.config import CLEAN_CSV, CLEAN_PARQUET, DIRECTORIES, QUALITY_TABLE, RAW_DATA_FILE, RAW_DIR
from bdtools.utils import ensure_directories, save_table


def find_raw_file(path: Path | None = None) -> Path:
    if path is not None:
        candidate = Path(path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"No existe el archivo indicado: {candidate}")

    if RAW_DATA_FILE.exists():
        return RAW_DATA_FILE

    candidates = []
    for pattern in ("*.xlsx", "*.xls", "*.csv"):
        candidates.extend(sorted(RAW_DIR.glob(pattern)))
    if not candidates:
        raise FileNotFoundError(f"No se encontró archivo de datos en {RAW_DIR}")
    return candidates[0]


def read_raw_data(path: Path | None = None) -> pd.DataFrame:
    source = find_raw_file(path)
    suffix = source.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source)
    if suffix == ".csv":
        for encoding in ("utf-8", "utf-8-sig", "latin1"):
            try:
                return pd.read_csv(source, encoding=encoding)
            except UnicodeDecodeError:
                continue
    raise ValueError(f"Formato no soportado: {source.suffix}")


def export_processed_data(df: pd.DataFrame) -> dict[str, Path]:
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False, encoding="utf-8-sig")
    df.to_parquet(CLEAN_PARQUET, index=False)
    return {"csv": CLEAN_CSV, "parquet": CLEAN_PARQUET}


def run_etl(path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_directories(DIRECTORIES)
    raw_df = read_raw_data(path)
    clean_df = clean_dataset(raw_df)
    quality = quality_report(raw_df, clean_df)
    export_processed_data(clean_df)
    save_table(quality, QUALITY_TABLE)
    return clean_df, quality
