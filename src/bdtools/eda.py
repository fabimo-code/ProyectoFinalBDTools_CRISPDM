from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bdtools.config import (
    CATEGORICAL_FREQUENCIES_TABLE,
    CATEGORICAL_REPORT_COLUMNS,
    COORDINATES_SUMMARY_TABLE,
    DROPPED_COLUMNS_TABLE,
    FIG_CASOS_ANIO,
    FIG_COLUMNAS_ELIMINADAS_NULOS,
    FIG_CONFLICT_2000_2010,
    FIG_COORDENADAS_EXTRAIDAS,
    FIG_CORRELACIONES_HECHOS,
    FIG_DEPARTAMENTOS,
    FIG_DESCRIPTIVE_PANEL,
    FIG_DISTRIBUCION_DECADA,
    FIG_EVOLUCION_HECHOS,
    FIG_FORMA_VINCULACION,
    FIG_FRECUENCIAS_CATEGORICAS,
    FIG_GEO_CASOS_VICTIMAS,
    FIG_GEO_RESP_2000_2010,
    FIG_HECHOS_DISTRIBUTION,
    FIG_MAPA_NULOS,
    FIG_MODALIDAD,
    FIG_MODALIDAD_DETAIL,
    FIG_MODALIDAD_FRECUENCIA_PCT,
    FIG_OUTLIERS_DEPARTAMENTO,
    FIG_OUTLIERS_VICTIMAS,
    FIG_PORCENTAJE_NULOS,
    FIG_REGIONES,
    FIG_RESPONSABLE,
    FIG_RESPONSABLE_DETAIL,
    FIG_RESP_DEPARTAMENTOS,
    FIG_RESP_MODALIDAD,
    FIG_TEMPORAL_ACTOR,
    FIG_TEMPORAL_DECADE_MONTH,
    FIG_TEMPORAL_EVOLUCION,
    FIG_TIPO_VINCULACION,
    FIG_TOP_DEPARTAMENTOS,
    FIG_VICTIMAS_BIVARIADO,
    FIG_VICTIMS_DISTRIBUTION,
    HECHOS_COLUMNS,
    HECHOS_LABELS,
    IMPLICIT_DATE_NULLS_TABLE,
    LAT_COL,
    LON_COL,
    NULLS_SUMMARY_TABLE,
    OUTLIERS_TABLE,
    SUMMARY_TABLE,
    TABLES_DIR,
    TEXT_COLUMNS,
    TEXT_STANDARDIZATION_TABLE,
    TOP_N,
    VICTIMS_COL,
)
from bdtools.utils import save_table

PALETTE = [
    "#C0392B", "#E67E22", "#F1C40F", "#27AE60", "#2980B9",
    "#8E44AD", "#16A085", "#D35400", "#7F8C8D", "#2C3E50",
]
RED = "#C0392B"
BLUE = "#2980B9"
DARK = "#1C1C1C"


def _style_axes(ax):
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")


def _save_fig(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _safe_text(series: pd.Series) -> pd.Series:
    return series.astype("object").where(pd.notna(series), "SIN INFORMACION").astype(str)


def _available(columns: list[str], df: pd.DataFrame) -> list[str]:
    return [c for c in columns if c in df.columns]


def _victims(df: pd.DataFrame) -> pd.Series:
    if VICTIMS_COL not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[VICTIMS_COL], errors="coerce")


def _existing_hechos(df: pd.DataFrame) -> list[str]:
    return [c for c in HECHOS_COLUMNS if c in df.columns]


def _binary_hechos(df: pd.DataFrame) -> pd.DataFrame:
    cols = _existing_hechos(df)
    if not cols:
        return pd.DataFrame(index=df.index)
    return df[cols].apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).astype(float))


def _top_values(df: pd.DataFrame, column: str, n: int = TOP_N, exclude: tuple[str, ...] = ()) -> pd.Index:
    if column not in df.columns:
        return pd.Index([])
    s = _safe_text(df[column])
    if exclude:
        s = s[~s.isin(exclude)]
    return s.value_counts().head(n).index


def frequency_table(df: pd.DataFrame, column: str, top_n: int | None = None) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=["variable", "categoria", "casos", "porcentaje"])
    series = _safe_text(df[column])
    counts = series.value_counts(dropna=False)
    if top_n:
        counts = counts.head(top_n)
    table = counts.rename_axis("categoria").reset_index(name="casos")
    table.insert(0, "variable", column)
    table["porcentaje"] = (table["casos"] / len(df) * 100).round(2)
    return table


def general_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"indicador": "registros", "valor": int(len(df))},
        {"indicador": "variables", "valor": int(df.shape[1])},
        {"indicador": "duplicados", "valor": int(df.duplicated().sum())},
    ]
    victims = _victims(df)
    if not victims.empty:
        rows.extend([
            {"indicador": "victimas_total", "valor": float(victims.sum())},
            {"indicador": "victimas_media", "valor": float(victims.mean())},
            {"indicador": "victimas_max", "valor": float(victims.max())},
        ])
    for col in ["anio_valido", "departamento", "region", "modalidad", "presunto_responsable"]:
        if col in df.columns:
            rows.append({"indicador": f"{col}_unicos", "valor": int(df[col].nunique(dropna=True))})
    return pd.DataFrame(rows)


def build_missing_reports(df: pd.DataFrame, quality: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    outputs = {}
    if quality is not None and not quality.empty and "nulos_raw_pct" in quality.columns:
        nulls = quality[["variable", "nulos_raw_pct", "nulos_limpio_pct", "tipo_limpio", "valores_unicos_limpio"]].copy()
        nulls = nulls.rename(columns={"nulos_raw_pct": "pct_nulos_raw", "nulos_limpio_pct": "pct_nulos_limpio"})
        nulls = nulls.sort_values("pct_nulos_raw", ascending=False)
    else:
        nulls = pd.DataFrame({
            "variable": df.columns,
            "pct_nulos_raw": df.isna().mean().values * 100,
            "pct_nulos_limpio": df.isna().mean().values * 100,
            "tipo_limpio": [str(t) for t in df.dtypes],
            "valores_unicos_limpio": [df[c].nunique(dropna=True) for c in df.columns],
        }).round(2)
    save_table(nulls, NULLS_SUMMARY_TABLE)
    outputs["mapa_nulos_resumen"] = nulls

    dropped = nulls[nulls["tipo_limpio"].eq("eliminada") | (nulls["pct_nulos_raw"] > 90)].copy()
    save_table(dropped, DROPPED_COLUMNS_TABLE)
    outputs["columnas_eliminadas_nulos"] = dropped

    fig, ax = plt.subplots(figsize=(12, max(5, min(14, len(nulls) * 0.28))))
    data = nulls.sort_values("pct_nulos_raw", ascending=True).tail(35)
    colors = [RED if v > 90 else ("#E67E22" if v > 50 else BLUE) for v in data["pct_nulos_raw"].fillna(0)]
    ax.barh(data["variable"], data["pct_nulos_raw"], color=colors)
    ax.axvline(90, color=RED, linestyle="--", linewidth=1.2, label="Umbral 90%")
    ax.axvline(50, color="#E67E22", linestyle=":", linewidth=1.1, label="Referencia 50%")
    ax.set_title("Porcentaje de valores nulos por variable", fontweight="bold")
    ax.set_xlabel("% de nulos")
    ax.legend(loc="lower right")
    _style_axes(ax)
    for y, v in enumerate(data["pct_nulos_raw"].fillna(0)):
        if v > 0:
            ax.text(v + 0.6, y, f"{v:.1f}%", va="center", fontsize=7)
    fig.tight_layout()
    _save_fig(fig, FIG_PORCENTAJE_NULOS)

    sample = df.iloc[: min(len(df), 500), : min(df.shape[1], 45)].isna().T.astype(int)
    fig, ax = plt.subplots(figsize=(13, max(4, min(12, sample.shape[0] * 0.24))))
    ax.imshow(sample, aspect="auto", interpolation="nearest", cmap="Greys")
    ax.set_title("Mapa de nulos del dataset limpio", fontweight="bold")
    ax.set_xlabel("Registros de muestra")
    ax.set_ylabel("Variables")
    ax.set_yticks(range(sample.shape[0]))
    ax.set_yticklabels(sample.index, fontsize=7)
    fig.tight_layout()
    _save_fig(fig, FIG_MAPA_NULOS)

    fig, ax = plt.subplots(figsize=(10, max(3.5, min(8, len(dropped) * 0.35 + 2))))
    if dropped.empty:
        ax.text(0.5, 0.5, "No hay columnas eliminadas por nulos > 90%", ha="center", va="center", fontsize=11)
        ax.axis("off")
    else:
        dd = dropped.sort_values("pct_nulos_raw", ascending=True)
        ax.barh(dd["variable"], dd["pct_nulos_raw"], color=RED)
        ax.set_title("Columnas eliminadas o candidatas por más de 90% de nulos", fontweight="bold")
        ax.set_xlabel("% de nulos")
        _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, FIG_COLUMNAS_ELIMINADAS_NULOS)
    return outputs


def build_cleaning_audit_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs = {}
    rows = []
    for raw_col, valid_col in [("anio", "anio_valido"), ("mes", "mes_valido"), ("dia", "dia_valido")]:
        if raw_col in df.columns and valid_col in df.columns:
            raw = pd.to_numeric(df[raw_col], errors="coerce")
            invalid = df[valid_col].isna()
            rows.append({
                "variable_original": raw_col,
                "variable_valida": valid_col,
                "nulos_implicitos_o_invalidos": int(invalid.sum()),
                "porcentaje": round(invalid.mean() * 100, 2),
                "ceros_originales": int((raw == 0).sum()),
            })
    implicit = pd.DataFrame(rows)
    save_table(implicit, IMPLICIT_DATE_NULLS_TABLE)
    outputs["nulos_implicitos_fechas"] = implicit

    text_rows = []
    for col in _available(TEXT_COLUMNS, df):
        s = _safe_text(df[col])
        text_rows.append({
            "variable": col,
            "categorias": int(s.nunique()),
            "sin_informacion": int(s.eq("SIN INFORMACION").sum()),
            "porcentaje_sin_informacion": round(s.eq("SIN INFORMACION").mean() * 100, 2),
        })
    text_table = pd.DataFrame(text_rows)
    save_table(text_table, TEXT_STANDARDIZATION_TABLE)
    outputs["estandarizacion_textos_categoricos"] = text_table

    coord_rows = []
    if LAT_COL in df.columns and LON_COL in df.columns:
        valid = df[LAT_COL].notna() & df[LON_COL].notna()
        coord_rows.append({
            "indicador": "coordenadas_validas",
            "valor": int(valid.sum()),
            "porcentaje": round(valid.mean() * 100, 2),
        })
        coord_rows.append({
            "indicador": "coordenadas_no_validas",
            "valor": int((~valid).sum()),
            "porcentaje": round((~valid).mean() * 100, 2),
        })
        fig, ax = plt.subplots(figsize=(7, 6))
        if valid.any():
            ax.scatter(df.loc[valid, LON_COL], df.loc[valid, LAT_COL], s=8, alpha=0.35, color=BLUE)
        ax.set_title("Coordenadas extraídas desde latitud-longitud", fontweight="bold")
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        _style_axes(ax)
        fig.tight_layout()
        _save_fig(fig, FIG_COORDENADAS_EXTRAIDAS)
    coords = pd.DataFrame(coord_rows)
    save_table(coords, COORDINATES_SUMMARY_TABLE)
    outputs["coordenadas_extraidas"] = coords
    return outputs


def build_categorical_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs = {}
    all_tables = []
    for col in _available(CATEGORICAL_REPORT_COLUMNS, df):
        table = frequency_table(df, col)
        outputs[f"frecuencia_{col}"] = table
        save_table(table, TABLES_DIR / f"frecuencia_{col}.csv")
        all_tables.append(table)
    combined = pd.concat(all_tables, ignore_index=True) if all_tables else pd.DataFrame()
    save_table(combined, CATEGORICAL_FREQUENCIES_TABLE)
    outputs["frecuencias_categoricas"] = combined
    return outputs


def _plot_count_bar(df: pd.DataFrame, column: str, path, title: str, top_n: int = TOP_N, horizontal: bool = True, exclude: tuple[str, ...] = ()) -> None:
    if column not in df.columns:
        return
    s = _safe_text(df[column])
    if exclude:
        s = s[~s.isin(exclude)]
    counts = s.value_counts().head(top_n)
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(11, max(4.5, min(8, len(counts) * 0.42))))
    if horizontal:
        data = counts.sort_values()
        ax.barh(data.index, data.values, color=PALETTE[: len(data)] if len(data) <= len(PALETTE) else BLUE)
        ax.set_xlabel("Nº de casos")
        for i, v in enumerate(data.values):
            ax.text(v + max(data.values) * 0.01, i, f"{int(v):,}", va="center", fontsize=8)
    else:
        ax.bar(counts.index, counts.values, color=PALETTE[: len(counts)] if len(counts) <= len(PALETTE) else BLUE)
        ax.set_ylabel("Nº de casos")
        ax.tick_params(axis="x", rotation=20)
        for x, v in enumerate(counts.values):
            ax.text(x, v + max(counts.values) * 0.01, f"{int(v):,}", ha="center", fontsize=8)
    ax.set_title(title, fontweight="bold")
    _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, path)


def plot_core_categorical_visuals(df: pd.DataFrame) -> None:
    _plot_count_bar(df, "departamento", FIG_TOP_DEPARTAMENTOS, "Top departamentos", top_n=TOP_N)
    _plot_count_bar(df, "modalidad", FIG_MODALIDAD, "Casos por modalidad", top_n=TOP_N)
    _plot_count_bar(df, "presunto_responsable", FIG_RESPONSABLE, "Casos por presunto responsable", top_n=TOP_N)
    _plot_count_bar(df, "tipo_vinculacion", FIG_TIPO_VINCULACION, "Tipo de vinculación", top_n=TOP_N)
    _plot_count_bar(df, "forma_vinculacion", FIG_FORMA_VINCULACION, "Forma de vinculación", top_n=TOP_N)
    _plot_count_bar(df, "departamento", FIG_DEPARTAMENTOS, "Departamentos", top_n=20)
    _plot_count_bar(df, "region", FIG_REGIONES, "Regiones", top_n=20)

    if "decada" in df.columns:
        counts = _safe_text(df["decada"]).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(counts.index, counts.values, color=RED)
        ax.set_title("Distribución por década", fontweight="bold")
        ax.set_xlabel("Década")
        ax.set_ylabel("Nº de casos")
        ax.tick_params(axis="x", rotation=20)
        _style_axes(ax)
        fig.tight_layout()
        _save_fig(fig, FIG_DISTRIBUCION_DECADA)

    if "modalidad" in df.columns:
        table = frequency_table(df, "modalidad")
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].bar(table["categoria"], table["casos"], color=PALETTE[: len(table)])
        axes[0].set_title("Modalidad: frecuencia", fontweight="bold")
        axes[0].set_ylabel("Nº de casos")
        axes[0].tick_params(axis="x", rotation=18)
        for i, v in enumerate(table["casos"]):
            axes[0].text(i, v + table["casos"].max() * 0.01, f"{int(v):,}", ha="center", fontsize=8)
        axes[1].pie(table["casos"], labels=table["categoria"], autopct="%1.1f%%", colors=PALETTE[: len(table)], startangle=130)
        axes[1].set_title("Modalidad: % del total", fontweight="bold")
        for ax in axes:
            _style_axes(ax)
        fig.tight_layout()
        _save_fig(fig, FIG_MODALIDAD_FRECUENCIA_PCT)

    cols = [c for c in ["modalidad", "tipo_vinculacion", "forma_vinculacion", "departamento", "region", "presunto_responsable"] if c in df.columns]
    if cols:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        for ax, col in zip(axes, cols):
            counts = _safe_text(df[col]).value_counts().head(8).sort_values()
            ax.barh(counts.index, counts.values, color=BLUE)
            ax.set_title(col.replace("_", " ").title(), fontweight="bold")
            _style_axes(ax)
        for ax in axes[len(cols):]:
            ax.axis("off")
        fig.suptitle("Frecuencia de variables categóricas clave", fontsize=14, fontweight="bold")
        fig.tight_layout()
        _save_fig(fig, FIG_FRECUENCIAS_CATEGORICAS)


def plot_cases_by_year(df: pd.DataFrame) -> None:
    if "anio_valido" not in df.columns:
        return
    data = df.dropna(subset=["anio_valido"]).groupby("anio_valido").size().reset_index(name="casos")
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(data["anio_valido"], data["casos"], marker="o", linewidth=1.5, color=RED)
    ax.set_title("Casos por año", fontweight="bold")
    ax.set_xlabel("Año")
    ax.set_ylabel("Casos")
    _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, FIG_CASOS_ANIO)


def plot_descriptive_panel(df: pd.DataFrame) -> None:
    victims = _victims(df)
    hechos = _binary_hechos(df)
    if victims.empty or hechos.empty:
        return
    total_hechos = df["total_hechos"] if "total_hechos" in df.columns else hechos.sum(axis=1)
    dist_hechos = total_hechos.value_counts().sort_index()
    resumen_hechos = hechos.sum().rename("Total Registrados").to_frame()
    resumen_hechos.index = [HECHOS_LABELS.get(c, c) for c in resumen_hechos.index]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Estadísticas Descriptivas — Conflicto Armado Colombia", fontsize=14, fontweight="bold")

    ax = axes[0]
    colors = ["#27AE60" if i == 0 else RED for i in range(len(dist_hechos))]
    ax.bar(dist_hechos.index.astype(str), dist_hechos.values, color=colors, edgecolor="white")
    ax.set_title("Hechos Simultáneos por Caso", fontweight="bold")
    ax.set_xlabel("Nº de Hechos")
    ax.set_ylabel("Nº de Casos")
    for x, y in zip(range(len(dist_hechos)), dist_hechos.values):
        ax.text(x, y + max(dist_hechos.values) * 0.01, f"{int(y):,}", ha="center", fontsize=8)
    _style_axes(ax)

    ax = axes[1]
    h = resumen_hechos.sort_values("Total Registrados")
    ax.barh(h.index, h["Total Registrados"], color=plt.cm.Reds(np.linspace(0.4, 0.9, len(h))), edgecolor="white")
    ax.set_title("Total registrado por Hecho Victimizante", fontweight="bold")
    ax.set_xlabel("Total de Registros")
    for i, v in enumerate(h["Total Registrados"]):
        ax.text(v + max(h["Total Registrados"]) * 0.015, i, f"{int(v):,}", va="center", fontsize=8)
    _style_axes(ax)

    ax = axes[2]
    if "presunto_responsable" in df.columns:
        top5 = _top_values(df, "presunto_responsable", 5)
        grupos = [victims[_safe_text(df["presunto_responsable"]) == r].dropna().values for r in top5]
        grupos = [g for g in grupos if len(g) > 0]
        labels = [r for r in top5 if len(victims[_safe_text(df["presunto_responsable"]) == r].dropna()) > 0]
        if grupos:
            bp = ax.boxplot(grupos, patch_artist=True, medianprops=dict(color="white", linewidth=2))
            for patch, color in zip(bp["boxes"], PALETTE):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels([r.replace(" ", "\n") for r in labels], fontsize=7)
    ax.set_title("Víctimas por Caso según Responsable", fontweight="bold")
    ax.set_ylabel("Total de Víctimas")
    _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, FIG_DESCRIPTIVE_PANEL)


def plot_victims_distribution(df: pd.DataFrame) -> None:
    victims = _victims(df).dropna()
    if victims.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Distribución — Total de Víctimas por Caso", fontsize=14, fontweight="bold")
    axes[0].hist(victims[victims <= 10], bins=range(1, 12), color=RED, alpha=0.8, edgecolor="white")
    axes[0].set_title("Histograma (casos con ≤ 10 víctimas)", fontweight="bold")
    axes[0].set_xlabel("Total de Víctimas")
    axes[0].set_ylabel("Frecuencia")
    axes[1].boxplot(victims.values, vert=True, patch_artist=True, boxprops=dict(facecolor=RED, alpha=0.35))
    axes[1].set_title("Boxplot — Total de Víctimas por Caso", fontweight="bold")
    axes[1].set_ylabel("Total de Víctimas")
    for ax in axes:
        _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, FIG_VICTIMS_DISTRIBUTION)


def plot_hechos_distribution(df: pd.DataFrame) -> None:
    hechos = _binary_hechos(df)
    if hechos.empty:
        return
    n = len(hechos.columns)
    rows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(18, rows * 4))
    axes = np.array(axes).ravel()
    fig.suptitle("Distribución de Hechos Victimizantes", fontsize=14, fontweight="bold")
    for ax, col, color in zip(axes, hechos.columns, PALETTE * 3):
        vc = hechos[col].value_counts().sort_index()
        ax.bar(vc.index.astype(str), vc.values, color=color, alpha=0.85)
        ax.set_title(HECHOS_LABELS.get(col, col), fontweight="bold", fontsize=9)
        ax.set_xlabel("Valor registrado")
        ax.set_ylabel("Frecuencia")
        for i, v in enumerate(vc.values):
            ax.text(i, v + max(vc.values) * 0.01, f"{int(v):,}", ha="center", fontsize=7)
        _style_axes(ax)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    _save_fig(fig, FIG_HECHOS_DISTRIBUTION)


def plot_responsable_detail(df: pd.DataFrame) -> None:
    if "presunto_responsable" not in df.columns:
        return
    s = _safe_text(df["presunto_responsable"])
    all_counts = s.value_counts().head(10)
    detail_counts = s[~s.isin(["DESCONOCIDO", "SIN INFORMACION"])].value_counts().head(10)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Análisis Univariado — Presunto Responsable", fontsize=14, fontweight="bold")
    for ax, counts, title, color in [(axes[0], all_counts, "Frecuencia por Presunto Responsable", RED), (axes[1], detail_counts, 'Sin "DESCONOCIDO" — Detalle de Actores', BLUE)]:
        data = counts.sort_values()
        ax.barh(data.index, data.values, color=color, alpha=0.75)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Nº de Casos")
        for i, v in enumerate(data.values):
            ax.text(v + max(data.values) * 0.01, i, f"{int(v):,} ({v/len(df)*100:.1f}%)", va="center", fontsize=8)
        _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, FIG_RESPONSABLE_DETAIL)


def plot_modalidad_detail(df: pd.DataFrame) -> None:
    if "modalidad" not in df.columns:
        return
    s = _safe_text(df["modalidad"])
    counts = s.value_counts()
    detail = s[~s.isin(["DESCONOCIDA", "SIN INFORMACION"])].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Análisis Univariado — Modalidad de Vinculación", fontsize=14, fontweight="bold")
    axes[0].bar(counts.index, counts.values, color=PALETTE[: len(counts)], edgecolor="white")
    axes[0].set_title("Todas las Modalidades", fontweight="bold")
    axes[0].set_ylabel("Nº de Casos")
    axes[0].tick_params(axis="x", rotation=15)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + max(counts.values) * 0.01, f"{int(v):,}", ha="center", fontsize=8)
    if not detail.empty:
        axes[1].pie(detail.values, labels=detail.index, autopct="%1.1f%%", colors=PALETTE[: len(detail)], startangle=140)
    axes[1].set_title('Sin "DESCONOCIDA" — Proporción', fontweight="bold")
    for ax in axes:
        _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, FIG_MODALIDAD_DETAIL)


def plot_temporal_decade_month(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Análisis Univariado — Distribución Temporal", fontsize=14, fontweight="bold")
    if "decada" in df.columns:
        d = _safe_text(df["decada"])
        d = d[d != "SIN INFORMACION"].value_counts().sort_index()
        axes[0].bar(d.index, d.values, color=RED)
        axes[0].set_title("Casos por Década (con fecha válida)", fontweight="bold")
        axes[0].set_ylabel("Nº de Casos")
        axes[0].tick_params(axis="x", rotation=20)
        for i, v in enumerate(d.values):
            axes[0].text(i, v + max(d.values) * 0.01 if len(d) else 0, f"{int(v):,}", ha="center", fontsize=8)
    if "mes_valido" in df.columns:
        m = pd.to_numeric(df["mes_valido"], errors="coerce").dropna().astype(int).value_counts().sort_index()
        labels = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
        m = m.reindex(range(1, 13), fill_value=0)
        axes[1].bar(labels, m.values, color=BLUE, alpha=0.8)
        axes[1].set_title("Estacionalidad — Casos por Mes", fontweight="bold")
        axes[1].set_ylabel("Nº de Casos")
    for ax in axes:
        _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, FIG_TEMPORAL_DECADE_MONTH)


def plot_conflict_2000_2010(df: pd.DataFrame) -> None:
    if "anio_valido" not in df.columns:
        return
    victims = _victims(df)
    d = df.assign(_victimas=victims)
    d = d[d["anio_valido"].between(2000, 2010)]
    if d.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Análisis Detallado — Conflicto Armado 2000–2010", fontsize=14, fontweight="bold")
    cases = d.groupby("anio_valido").size()
    vict = d.groupby("anio_valido")["_victimas"].sum()
    axes[0, 0].plot(cases.index, cases.values, marker="o", color=RED)
    axes[0, 0].fill_between(cases.index.astype(float), cases.values, color=RED, alpha=0.18)
    axes[0, 0].set_title("Casos por Año", fontweight="bold")
    axes[0, 0].set_xlabel("Año"); axes[0, 0].set_ylabel("Nº de Casos")
    axes[0, 1].plot(vict.index, vict.values, marker="o", color=BLUE)
    axes[0, 1].fill_between(vict.index.astype(float), vict.values, color=BLUE, alpha=0.18)
    axes[0, 1].set_title("Total de Víctimas por Año", fontweight="bold")
    axes[0, 1].set_xlabel("Año"); axes[0, 1].set_ylabel("Total Víctimas")
    if "presunto_responsable" in d.columns:
        top = _top_values(d, "presunto_responsable", 4, exclude=("DESCONOCIDO", "SIN INFORMACION"))
        for r, color in zip(top, PALETTE):
            s = d[_safe_text(d["presunto_responsable"]) == r].groupby("anio_valido").size()
            axes[1, 0].plot(s.index, s.values, marker="o", label=r, color=color)
        axes[1, 0].set_title("Casos por Responsable (sin Desconocido)", fontweight="bold")
        axes[1, 0].set_xlabel("Año"); axes[1, 0].set_ylabel("Nº de Casos")
        axes[1, 0].legend(fontsize=8)
    hechos = _binary_hechos(d)
    if not hechos.empty:
        top_h = hechos.sum().sort_values(ascending=False).head(3).index
        for col, color in zip(top_h, PALETTE):
            s = d.assign(_h=hechos[col]).groupby("anio_valido")["_h"].sum()
            axes[1, 1].plot(s.index, s.values, marker="o", label=HECHOS_LABELS.get(col, col), color=color)
        axes[1, 1].set_title("Top 3 Hechos Victimizantes por Año", fontweight="bold")
        axes[1, 1].set_xlabel("Año"); axes[1, 1].set_ylabel("Total Registrado")
        axes[1, 1].legend(fontsize=8)
    for ax in axes.ravel():
        _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, FIG_CONFLICT_2000_2010)


def plot_geo_resp_2000_2010(df: pd.DataFrame) -> None:
    if "anio_valido" not in df.columns:
        return
    d = df[df["anio_valido"].between(2000, 2010)].copy()
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Distribución Geográfica y Responsables — 2000–2010", fontsize=14, fontweight="bold")
    if "departamento" in d.columns:
        dep = _safe_text(d["departamento"])
        dep = dep[dep != "SIN INFORMACION"].value_counts().head(10).sort_values()
        axes[0].barh(dep.index, dep.values, color=RED, alpha=0.75)
        axes[0].set_title("Top 10 Departamentos", fontweight="bold")
        axes[0].set_xlabel("Nº de Casos")
        for i, v in enumerate(dep.values):
            axes[0].text(v + max(dep.values) * 0.01, i, f"{int(v):,}", va="center", fontsize=8)
    if "presunto_responsable" in d.columns:
        resp = _safe_text(d["presunto_responsable"])
        resp = resp[~resp.isin(["DESCONOCIDO", "SIN INFORMACION"])].value_counts().head(6)
        if not resp.empty:
            axes[1].pie(resp.values, labels=resp.index, autopct="%1.1f%%", colors=PALETTE[: len(resp)], startangle=140)
        axes[1].set_title("Responsables Identificados\n(sin Desconocido)", fontweight="bold")
    for ax in axes:
        _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, FIG_GEO_RESP_2000_2010)


def plot_correlations(df: pd.DataFrame) -> None:
    hechos = _binary_hechos(df)
    victims = _victims(df)
    if hechos.empty or victims.empty:
        return
    corr = hechos.corr()
    corr_v = hechos.apply(lambda x: x.corr(victims)).sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.suptitle("Análisis de Correlaciones — Hechos Victimizantes", fontsize=14, fontweight="bold")
    labels = [HECHOS_LABELS.get(c, c) for c in corr.columns]
    mask = np.triu(np.ones_like(corr, dtype=bool))
    matrix = corr.mask(mask)
    im = axes[0].imshow(matrix, cmap="YlGn", vmin=0, vmax=max(0.5, np.nanmax(matrix.values) if np.isfinite(matrix.values).any() else 0.5))
    axes[0].set_xticks(range(len(labels))); axes[0].set_yticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8); axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_title("Correlación entre Hechos Victimizantes", fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if pd.notna(matrix.iloc[i, j]):
                axes[0].text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes[0], shrink=0.75)
    y = np.arange(len(corr_v))
    colors = [RED if v >= 0 else BLUE for v in corr_v.values]
    axes[1].barh([HECHOS_LABELS.get(c, c) for c in corr_v.index], corr_v.values, color=colors, alpha=0.75)
    axes[1].axvline(0, color=DARK, linewidth=0.8)
    axes[1].set_title("Correlación de cada Hecho\ncon Total de Víctimas", fontweight="bold")
    axes[1].set_xlabel("Coeficiente de Correlación")
    for i, v in enumerate(corr_v.values):
        axes[1].text(v + (0.002 if v >= 0 else -0.002), i, f"{v:.3f}", va="center", fontsize=8)
    _style_axes(axes[1])
    fig.tight_layout()
    _save_fig(fig, FIG_CORRELACIONES_HECHOS)


def plot_resp_modalidad(df: pd.DataFrame) -> None:
    if not {"presunto_responsable", "modalidad"}.issubset(df.columns):
        return
    resp = _safe_text(df["presunto_responsable"])
    mod = _safe_text(df["modalidad"])
    mask = ~resp.isin(["DESCONOCIDO", "SIN INFORMACION"]) & ~mod.isin(["DESCONOCIDA", "SIN INFORMACION"])
    d = pd.DataFrame({"responsable": resp[mask], "modalidad": mod[mask]})
    if d.empty:
        return
    top_resp = d["responsable"].value_counts().head(8).index
    top_mod = d["modalidad"].value_counts().head(5).index
    tab = pd.crosstab(d.loc[d["responsable"].isin(top_resp), "responsable"], d.loc[d["responsable"].isin(top_resp), "modalidad"]).reindex(columns=top_mod, fill_value=0)
    prop = tab.div(tab.sum(axis=1), axis=0).fillna(0) * 100
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.suptitle("Responsable vs Modalidad de Vinculación", fontsize=14, fontweight="bold")
    im = axes[0].imshow(tab.values, cmap="YlOrRd")
    axes[0].set_xticks(range(len(tab.columns))); axes[0].set_yticks(range(len(tab.index)))
    axes[0].set_xticklabels(tab.columns, rotation=20, ha="right"); axes[0].set_yticklabels(tab.index)
    axes[0].set_title("Frecuencia Responsable × Modalidad\n(sin Desconocidos)", fontweight="bold")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            axes[0].text(j, i, str(int(tab.iloc[i, j])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[0], shrink=0.7)
    bottom = np.zeros(len(prop))
    for col, color in zip(prop.columns, PALETTE):
        axes[1].barh(prop.index, prop[col], left=bottom, label=col, color=color, alpha=0.85)
        bottom += prop[col].values
    axes[1].set_title("Proporción de Modalidad por Responsable\n(sin Desconocidos)", fontweight="bold")
    axes[1].set_xlabel("% de Casos")
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    _style_axes(axes[1])
    fig.tight_layout()
    _save_fig(fig, FIG_RESP_MODALIDAD)


def plot_resp_departamentos(df: pd.DataFrame) -> None:
    if not {"presunto_responsable", "departamento"}.issubset(df.columns):
        return
    d = pd.DataFrame({"responsable": _safe_text(df["presunto_responsable"]), "departamento": _safe_text(df["departamento"])})
    d = d[~d["responsable"].isin(["DESCONOCIDO", "SIN INFORMACION"]) & ~d["departamento"].isin(["SIN INFORMACION"])]
    if d.empty:
        return
    top_dep = d["departamento"].value_counts().head(10).index
    top_resp = d["responsable"].value_counts().head(4).index
    tab = pd.crosstab(d["departamento"], d["responsable"]).reindex(index=top_dep, columns=top_resp, fill_value=0)
    fig, ax = plt.subplots(figsize=(17, 7))
    x = np.arange(len(tab.index))
    width = 0.8 / max(1, len(tab.columns))
    for i, (col, color) in enumerate(zip(tab.columns, PALETTE)):
        ax.bar(x + i * width, tab[col].values, width=width, label=col, color=color, alpha=0.8)
    ax.set_xticks(x + width * (len(tab.columns) - 1) / 2)
    ax.set_xticklabels(tab.index, rotation=25, ha="right")
    ax.set_title("Casos por Departamento según Responsable\n(Top 10 deptos, Top 4 responsables)", fontweight="bold")
    ax.set_xlabel("Departamento")
    ax.set_ylabel("Nº de Casos")
    ax.legend(title="Responsable", fontsize=8)
    _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, FIG_RESP_DEPARTAMENTOS)


def plot_victims_bivariate(df: pd.DataFrame) -> None:
    victims = _victims(df)
    if victims.empty:
        return
    d = df.assign(_victimas=victims)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Total de Víctimas — Bivariado", fontsize=14, fontweight="bold")
    if "presunto_responsable" in d.columns:
        resp = _safe_text(d["presunto_responsable"])
        mask = ~resp.isin(["DESCONOCIDO", "SIN INFORMACION"])
        agg = d[mask].groupby(resp[mask])["_victimas"].sum().sort_values(ascending=False).head(10).sort_values()
        axes[0].barh(agg.index, agg.values, color=RED, alpha=0.75)
        axes[0].set_title("Total Víctimas por Responsable\n(sin Desconocido)", fontweight="bold")
        axes[0].set_xlabel("Total de Víctimas")
    if "modalidad" in d.columns:
        mod = _safe_text(d["modalidad"])
        mask = ~mod.isin(["DESCONOCIDA", "SIN INFORMACION"])
        agg = d[mask].groupby(mod[mask])["_victimas"].mean().sort_values(ascending=False)
        axes[1].bar(agg.index, agg.values, color=BLUE, alpha=0.75)
        axes[1].set_title("Promedio de Víctimas por Modalidad\n(sin Desconocida)", fontweight="bold")
        axes[1].set_ylabel("Promedio de Víctimas por Caso")
        axes[1].tick_params(axis="x", rotation=15)
        for i, v in enumerate(agg.values):
            axes[1].text(i, v + max(agg.values) * 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for ax in axes:
        _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, FIG_VICTIMAS_BIVARIADO)


def plot_temporal_evolution(df: pd.DataFrame) -> None:
    if "anio_valido" not in df.columns:
        return
    d = df.dropna(subset=["anio_valido"]).copy()
    if d.empty:
        return
    d["_victimas"] = _victims(d)
    by_year = d.groupby("anio_valido").agg(casos=("anio_valido", "size"), victimas=("_victimas", "sum")).sort_index()
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Análisis Temporal — Evolución del Conflicto Armado", fontsize=14, fontweight="bold")
    axes[0, 0].plot(by_year.index, by_year["casos"], color=RED, label="Casos")
    axes[0, 0].set_title("Casos vs Víctimas por Año", fontweight="bold")
    axes[0, 0].set_ylabel("Nº de Casos", color=RED)
    ax2 = axes[0, 0].twinx()
    ax2.plot(by_year.index, by_year["victimas"], color=BLUE, linestyle="--", label="Víctimas")
    ax2.set_ylabel("Total de Víctimas", color=BLUE)
    roll = by_year["casos"].rolling(3, min_periods=1).mean()
    axes[0, 1].bar(by_year.index, by_year["casos"], color=RED, alpha=0.25, label="Casos anuales")
    axes[0, 1].plot(by_year.index, roll, color=DARK, linewidth=2, label="Media móvil (3 años)")
    axes[0, 1].set_title("Tendencia con Media Móvil (3 años)", fontweight="bold")
    axes[0, 1].legend(fontsize=8)
    if "mes_valido" in d.columns:
        heat = pd.crosstab(d["anio_valido"], d["mes_valido"]).reindex(columns=range(1, 13), fill_value=0)
        heat = heat[heat.index >= 1990]
        if not heat.empty:
            im = axes[1, 0].imshow(heat.values, aspect="auto", cmap="YlOrRd")
            axes[1, 0].set_yticks(range(len(heat.index)))
            axes[1, 0].set_yticklabels(heat.index.astype(int), fontsize=7)
            axes[1, 0].set_xticks(range(12))
            axes[1, 0].set_xticklabels(["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"])
            axes[1, 0].set_title("Heatmap Año × Mes (desde 1990)", fontweight="bold")
            fig.colorbar(im, ax=axes[1, 0], shrink=0.75)
        m = d["mes_valido"].dropna().astype(int).value_counts().reindex(range(1, 13), fill_value=0)
        axes[1, 1].bar(["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"], m.values, color=PALETTE * 2)
        axes[1, 1].set_title("Estacionalidad — Casos por Mes\n(todos los años)", fontweight="bold")
        for i, v in enumerate(m.values):
            axes[1, 1].text(i, v + max(m.values) * 0.01, f"{int(v):,}", ha="center", fontsize=8)
    for ax in axes.ravel():
        _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, FIG_TEMPORAL_EVOLUCION)


def plot_temporal_actor(df: pd.DataFrame) -> None:
    if not {"anio_valido", "presunto_responsable"}.issubset(df.columns):
        return
    d = df.dropna(subset=["anio_valido"]).copy()
    d["responsable"] = _safe_text(d["presunto_responsable"])
    d = d[~d["responsable"].isin(["DESCONOCIDO", "SIN INFORMACION"])]
    if d.empty:
        return
    top = d["responsable"].value_counts().head(3).index
    tab = pd.crosstab(d["anio_valido"], d["responsable"]).reindex(columns=top, fill_value=0).sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Evolución Temporal por Actor Armado", fontsize=14, fontweight="bold")
    for col, color in zip(tab.columns, PALETTE):
        axes[0].plot(tab.index, tab[col], marker="o", label=col, color=color)
    axes[0].set_title("Casos por Año según Responsable\n(Top 3 identificados)", fontweight="bold")
    axes[0].set_xlabel("Año"); axes[0].set_ylabel("Nº de Casos"); axes[0].legend(fontsize=8)
    axes[1].stackplot(tab.index.astype(float), [tab[c].values for c in tab.columns], labels=tab.columns, colors=PALETTE[: len(tab.columns)], alpha=0.75)
    axes[1].set_title("Área Apilada — Casos por Actor Armado", fontweight="bold")
    axes[1].set_xlabel("Año"); axes[1].set_ylabel("Nº de Casos"); axes[1].legend(fontsize=8)
    for ax in axes:
        _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, FIG_TEMPORAL_ACTOR)


def plot_evolucion_hechos(df: pd.DataFrame) -> None:
    if "anio_valido" not in df.columns:
        return
    hechos = _binary_hechos(df)
    d = df.dropna(subset=["anio_valido"]).copy()
    hechos = hechos.loc[d.index]
    if hechos.empty or d.empty:
        return
    top = hechos.sum().sort_values(ascending=False).head(4).index
    fig, ax = plt.subplots(figsize=(17, 6))
    fig.suptitle("Evolución de Hechos Victimizantes por Año", fontsize=14, fontweight="bold")
    for col, color in zip(top, PALETTE):
        s = d.assign(_h=hechos[col]).groupby("anio_valido")["_h"].sum().sort_index()
        ax.plot(s.index, s.values, marker="o", label=HECHOS_LABELS.get(col, col), color=color)
    ax.set_title("Top 4 Hechos Victimizantes a lo Largo del Tiempo", fontweight="bold")
    ax.set_xlabel("Año"); ax.set_ylabel("Total Registrado")
    ax.legend(fontsize=8)
    _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, FIG_EVOLUCION_HECHOS)


def plot_geospatial_cases_victims(df: pd.DataFrame) -> None:
    if "departamento" not in df.columns:
        return
    victims = _victims(df)
    d = df.assign(_victimas=victims, _departamento=_safe_text(df["departamento"]))
    d = d[d["_departamento"] != "SIN INFORMACION"]
    if d.empty:
        return
    cases = d["_departamento"].value_counts().head(15).sort_values()
    vict = d.groupby("_departamento")["_victimas"].sum().sort_values(ascending=False).head(15).sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Distribución Geoespacial — Casos y Víctimas", fontsize=14, fontweight="bold")
    axes[0].barh(cases.index, cases.values, color=RED, alpha=0.75)
    axes[0].set_title("Top 15 Departamentos — Nº de Casos", fontweight="bold")
    axes[0].set_xlabel("Nº de Casos")
    axes[1].barh(vict.index, vict.values, color=BLUE, alpha=0.75)
    axes[1].set_title("Top 15 Departamentos — Total Víctimas", fontweight="bold")
    axes[1].set_xlabel("Total de Víctimas")
    for ax in axes:
        _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, FIG_GEO_CASOS_VICTIMAS)


def plot_outliers(df: pd.DataFrame) -> None:
    victims = _victims(df).dropna()
    if victims.empty:
        return
    q1, q3 = victims.quantile(0.25), victims.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    outlier_mask = _victims(df) > upper
    outliers = df.loc[outlier_mask].copy()
    outliers[VICTIMS_COL] = _victims(outliers)
    save_table(outliers, OUTLIERS_TABLE)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Detección de Outliers — Total de Víctimas por Caso", fontsize=14, fontweight="bold")
    axes[0].boxplot(victims.values, patch_artist=True, boxprops=dict(facecolor=RED, alpha=0.3), flierprops=dict(marker="o", markersize=3, alpha=0.3))
    axes[0].axhline(upper, color="#E67E22", linestyle="--", label=f"Límite IQR ({upper:.1f})")
    axes[0].set_title("Boxplot General — Víctimas por Caso", fontweight="bold")
    axes[0].set_ylabel("Total de Víctimas")
    axes[0].legend(fontsize=8)
    if "presunto_responsable" in df.columns:
        top = _top_values(df, "presunto_responsable", 4, exclude=("DESCONOCIDO", "SIN INFORMACION"))
        groups = [_victims(df[_safe_text(df["presunto_responsable"]) == r]).dropna().values for r in top]
        groups = [g for g in groups if len(g) > 0]
        if groups:
            axes[1].boxplot(groups, patch_artist=True, flierprops=dict(marker="o", markersize=3, alpha=0.35))
            axes[1].set_xticks(range(1, len(groups) + 1))
            axes[1].set_xticklabels([r.replace(" ", "\n") for r in top[: len(groups)]], fontsize=8)
    axes[1].set_title("Outliers por Presunto Responsable", fontweight="bold")
    axes[1].set_ylabel("Total de Víctimas")
    for ax in axes:
        _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, FIG_OUTLIERS_VICTIMAS)

    if "departamento" in outliers.columns and not outliers.empty:
        dep = _safe_text(outliers["departamento"]).value_counts().head(12).sort_values()
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.suptitle("Distribución de Outliers por Departamento", fontsize=14, fontweight="bold")
        ax.barh(dep.index, dep.values, color=RED, alpha=0.75)
        ax.set_title("Top 12 Departamentos con más Casos Extremos", fontweight="bold")
        ax.set_xlabel("Nº de Outliers")
        for i, v in enumerate(dep.values):
            ax.text(v + max(dep.values) * 0.02, i, f"{int(v):,}", va="center", fontsize=8)
        _style_axes(ax)
        fig.tight_layout()
        _save_fig(fig, FIG_OUTLIERS_DEPARTAMENTO)


def plot_all_extended_visuals(df: pd.DataFrame) -> None:
    plot_cases_by_year(df)
    plot_core_categorical_visuals(df)
    plot_descriptive_panel(df)
    plot_victims_distribution(df)
    plot_hechos_distribution(df)
    plot_responsable_detail(df)
    plot_modalidad_detail(df)
    plot_temporal_decade_month(df)
    plot_conflict_2000_2010(df)
    plot_geo_resp_2000_2010(df)
    plot_correlations(df)
    plot_resp_modalidad(df)
    plot_resp_departamentos(df)
    plot_victims_bivariate(df)
    plot_temporal_evolution(df)
    plot_temporal_actor(df)
    plot_evolucion_hechos(df)
    plot_geospatial_cases_victims(df)
    plot_outliers(df)


def build_eda_reports(df: pd.DataFrame, quality: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, pd.DataFrame] = {}

    summary = general_summary(df)
    save_table(summary, SUMMARY_TABLE)
    outputs["resumen_general"] = summary

    outputs.update(build_missing_reports(df, quality))
    outputs.update(build_cleaning_audit_tables(df))
    outputs.update(build_categorical_tables(df))
    plot_all_extended_visuals(df)
    return outputs
