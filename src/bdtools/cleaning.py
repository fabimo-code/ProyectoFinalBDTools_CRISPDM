import re
from typing import Iterable

import numpy as np
import pandas as pd

from bdtools.config import (
    COLUMN_ALIASES,
    DAY_COL,
    GEO_SOURCE_COL,
    HECHOS_COLUMNS,
    LAT_COL,
    LON_COL,
    MONTH_COL,
    NULL_THRESHOLD,
    TARGET_COL,
    TEXT_COLUMNS,
    VICTIMS_COL,
    YEAR_COL,
)
from bdtools.utils import normalize_name


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    normalized = [normalize_name(col) for col in result.columns]
    normalized = [COLUMN_ALIASES.get(col, col) for col in normalized]
    result.columns = _deduplicate_columns(normalized)
    return result


def _deduplicate_columns(columns: Iterable[str]) -> list[str]:
    counts: dict[str, int] = {}
    output: list[str] = []
    for col in columns:
        if col not in counts:
            counts[col] = 0
            output.append(col)
        else:
            counts[col] += 1
            output.append(f"{col}_{counts[col]}")
    return output


def drop_sparse_columns(df: pd.DataFrame, threshold: float = NULL_THRESHOLD) -> pd.DataFrame:
    null_rate = df.isna().mean()
    keep_cols = null_rate[null_rate <= threshold].index.tolist()
    return df.loc[:, keep_cols].copy()


def clean_temporal_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if YEAR_COL in result.columns:
        result["anio_valido"] = _valid_year(result[YEAR_COL])
    if MONTH_COL in result.columns:
        result["mes_valido"] = _valid_numeric(result[MONTH_COL], lower=1, upper=12)
    if DAY_COL in result.columns:
        result["dia_valido"] = _valid_numeric(result[DAY_COL], lower=1, upper=31)
    if "anio_valido" in result.columns:
        result["decada"] = result["anio_valido"].apply(_decade_label)
    return result


def _valid_numeric(series: pd.Series, lower: int, upper: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    values = values.round()
    values = values.where(values.between(lower, upper))
    return values.astype("Int64")


def _valid_year(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")

    # Algunos CSV llegan con años como 2.004 o 1.998 por separador de miles.
    # Se normalizan a 2004 y 1998 antes de validar el rango.
    scaled = values.where(~values.between(1, 3), values * 1000)
    scaled = scaled.round()
    scaled = scaled.where(scaled.between(1900, 2100))
    return scaled.astype("Int64")


def _decade_label(value) -> str:
    if pd.isna(value):
        return "SIN INFORMACION"
    start = int(value) // 10 * 10
    return f"{start}-{start + 9}"


def standardize_text(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    result = df.copy()
    cols = columns or TEXT_COLUMNS
    for col in cols:
        if col in result.columns:
            text = result[col].astype("string").str.strip().str.upper()
            text = text.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "NULL": pd.NA})
            result[col] = text.fillna("SIN INFORMACION")
    return result


def impute_region_from_department(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if {"departamento", "region"}.issubset(result.columns):
        valid = result[result["region"].notna() & (result["region"] != "SIN INFORMACION")]
        if not valid.empty:
            mode_by_dept = valid.groupby("departamento")["region"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
            missing = result["region"].isna() | (result["region"] == "SIN INFORMACION")
            result.loc[missing, "region"] = result.loc[missing, "departamento"].map(mode_by_dept).fillna("SIN INFORMACION")
    return result


def extract_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if LAT_COL not in result.columns:
        result[LAT_COL] = np.nan
    if LON_COL not in result.columns:
        result[LON_COL] = np.nan

    if GEO_SOURCE_COL in result.columns:
        coords = result[GEO_SOURCE_COL].apply(_parse_coordinate_pair)
        parsed = pd.DataFrame(coords.tolist(), index=result.index, columns=[LAT_COL, LON_COL])
        result[LAT_COL] = pd.to_numeric(result[LAT_COL], errors="coerce").fillna(parsed[LAT_COL])
        result[LON_COL] = pd.to_numeric(result[LON_COL], errors="coerce").fillna(parsed[LON_COL])
    else:
        result[LAT_COL] = pd.to_numeric(result[LAT_COL], errors="coerce")
        result[LON_COL] = pd.to_numeric(result[LON_COL], errors="coerce")

    valid = result[LAT_COL].between(-5, 14) & result[LON_COL].between(-83, -66)
    result.loc[~valid, [LAT_COL, LON_COL]] = np.nan
    return result


def _parse_coordinate_pair(value) -> tuple[float, float]:
    if pd.isna(value):
        return (np.nan, np.nan)
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", str(value))
    if len(numbers) < 2:
        return (np.nan, np.nan)
    first, second = float(numbers[0]), float(numbers[1])

    first_is_lat = -5 <= first <= 14
    first_is_lon = -83 <= first <= -66
    second_is_lat = -5 <= second <= 14
    second_is_lon = -83 <= second <= -66

    if first_is_lat and second_is_lon:
        return (first, second)
    if first_is_lon and second_is_lat:
        return (second, first)
    return (first, second)


def build_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if VICTIMS_COL in result.columns:
        victims = pd.to_numeric(result[VICTIMS_COL], errors="coerce")
        result[VICTIMS_COL] = victims
        result[TARGET_COL] = (victims.fillna(0) > 1).astype(int)

    existing_hechos = [col for col in HECHOS_COLUMNS if col in result.columns]
    if existing_hechos:
        binary = result[existing_hechos].apply(lambda col: col.map(_as_binary))
        result["total_hechos"] = binary.sum(axis=1)
    elif "total_hechos" not in result.columns:
        result["total_hechos"] = 0
    return result


def _as_binary(value) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, (int, float, np.number)):
        return int(value > 0)
    text = str(value).strip().upper()
    return int(text not in {"", "0", "NO", "N", "FALSO", "FALSE", "NAN", "SIN INFORMACION"})


def quality_report(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    raw = standardize_columns(raw_df)
    rows = []
    for col in sorted(set(raw.columns).union(clean_df.columns)):
        raw_null = raw[col].isna().mean() if col in raw.columns else np.nan
        clean_null = clean_df[col].isna().mean() if col in clean_df.columns else np.nan
        rows.append(
            {
                "variable": col,
                "tipo_limpio": str(clean_df[col].dtype) if col in clean_df.columns else "eliminada",
                "nulos_raw_pct": round(raw_null * 100, 2) if pd.notna(raw_null) else np.nan,
                "nulos_limpio_pct": round(clean_null * 100, 2) if pd.notna(clean_null) else np.nan,
                "valores_unicos_limpio": int(clean_df[col].nunique(dropna=True)) if col in clean_df.columns else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(["nulos_limpio_pct", "variable"], ascending=[False, True])


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    result = standardize_columns(df)
    result = drop_sparse_columns(result)
    result = clean_temporal_columns(result)
    result = standardize_text(result)
    result = impute_region_from_department(result)
    result = extract_coordinates(result)
    result = build_derived_variables(result)
    return result
