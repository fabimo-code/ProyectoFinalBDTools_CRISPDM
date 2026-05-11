from __future__ import annotations

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import bootstrap, chi2_contingency, kruskal, mannwhitneyu, normaltest, spearmanr

from bdtools.config import (
    ADVANCED_GROUPS_TABLE,
    ADVANCED_INFERENCE_TABLE,
    BOOTSTRAP_TABLE,
    CHI2_TABLE,
    CHI2_VARIABLES,
    DESCRIPTIVE_TABLE,
    FIG_INFERENTIAL_ADVANCED,
    HECHOS_COLUMNS,
    HECHOS_LABELS,
    RANDOM_STATE,
    SPEARMAN_TABLE,
    TARGET_COL,
    VICTIMS_COL,
)
from bdtools.utils import save_table

PALETTE = ["#C0392B", "#E67E22", "#F1C40F", "#27AE60", "#2980B9", "#8E44AD", "#16A085", "#D35400"]
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


def _victims(df: pd.DataFrame) -> pd.Series:
    if VICTIMS_COL not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[VICTIMS_COL], errors="coerce")


def _existing_hechos(df: pd.DataFrame) -> list[str]:
    return [c for c in HECHOS_COLUMNS if c in df.columns]


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=["number", "Int64", "Float64"])
    if numeric.empty:
        return pd.DataFrame()
    summary = numeric.describe().T.reset_index().rename(columns={"index": "variable"})
    return summary.round(3)


def proportion_confidence_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = successes / n
    error = z * math.sqrt(p * (1 - p) / n)
    return (round(p, 4), round(max(0, p - error), 4), round(min(1, p + error), 4))


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"indicador": "casos", "valor": len(df)},
        {"indicador": "variables", "valor": df.shape[1]},
    ]
    victims = _victims(df)
    if not victims.empty:
        rows.extend([
            {"indicador": "victimas_total", "valor": victims.sum()},
            {"indicador": "victimas_media", "valor": victims.mean()},
            {"indicador": "victimas_mediana", "valor": victims.median()},
            {"indicador": "victimas_desviacion", "valor": victims.std()},
            {"indicador": "victimas_min", "valor": victims.min()},
            {"indicador": "victimas_max", "valor": victims.max()},
        ])
    if TARGET_COL in df.columns:
        target = pd.to_numeric(df[TARGET_COL], errors="coerce").dropna().astype(int)
        p, lower, upper = proportion_confidence_interval(int(target.sum()), int(target.count()))
        rows.extend([
            {"indicador": "alto_impacto_proporcion", "valor": p},
            {"indicador": "alto_impacto_ic95_inf", "valor": lower},
            {"indicador": "alto_impacto_ic95_sup", "valor": upper},
        ])
    return pd.DataFrame(rows).round(4)


def chi_square_tests(df: pd.DataFrame, variables: list[str] | None = None, target: str = TARGET_COL) -> pd.DataFrame:
    variables = variables or CHI2_VARIABLES
    rows = []
    if target not in df.columns:
        return pd.DataFrame(columns=["variable", "chi2", "p_value", "dof", "cramers_v", "decision"])

    for variable in variables:
        if variable not in df.columns:
            continue
        subset = df[[variable, target]].dropna().copy()
        if subset.empty or subset[variable].nunique() < 2 or subset[target].nunique() < 2:
            continue
        contingency = pd.crosstab(subset[variable], subset[target])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue
        chi2, p_value, dof, _ = chi2_contingency(contingency)
        n = contingency.to_numpy().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = math.sqrt((chi2 / n) / min_dim) if n > 0 and min_dim > 0 else np.nan
        rows.append({
            "variable": variable,
            "chi2": round(chi2, 4),
            "p_value": round(p_value, 6),
            "dof": int(dof),
            "cramers_v": round(cramers_v, 4) if pd.notna(cramers_v) else np.nan,
            "decision": "asociacion" if p_value < 0.05 else "sin_asociacion",
        })
    return pd.DataFrame(rows).sort_values("p_value") if rows else pd.DataFrame(rows)


def _bootstrap_mean(values: np.ndarray, n_resamples: int = 1000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (np.nan, np.nan, np.nan)
    result = bootstrap(
        (values,),
        statistic=np.mean,
        confidence_level=0.95,
        n_resamples=n_resamples,
        random_state=RANDOM_STATE,
        method="percentile",
    )
    return (float(np.mean(values)), float(result.confidence_interval.low), float(result.confidence_interval.high))


def advanced_inference(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    victims = _victims(df).dropna()
    rows = []
    group_rows = []
    spearman_rows = []
    boot_rows = []

    if len(victims) >= 8:
        stat, p = normaltest(victims.values)
        rows.append({
            "prueba": "normalidad_dagostino_pearson",
            "variable": VICTIMS_COL,
            "estadistico": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "decision": "no_normal" if p < 0.05 else "normal",
        })

    if "presunto_responsable" in df.columns and not victims.empty:
        resp = _safe_text(df["presunto_responsable"])
        top_resp = resp[~resp.isin(["DESCONOCIDO", "SIN INFORMACION"])].value_counts().head(5).index
        grupos = [victims[resp.loc[victims.index] == r].dropna().values for r in top_resp]
        grupos = [g for g in grupos if len(g) > 1]
        labels = [r for r in top_resp if len(victims[resp.loc[victims.index] == r].dropna()) > 1]
        if len(grupos) >= 2:
            stat, p = kruskal(*grupos)
            rows.append({
                "prueba": "kruskal_responsable",
                "variable": "presunto_responsable",
                "estadistico": round(float(stat), 4),
                "p_value": round(float(p), 6),
                "decision": "diferencias" if p < 0.05 else "sin_diferencias",
            })
            for label, g in zip(labels, grupos):
                group_rows.append({
                    "prueba": "kruskal_responsable",
                    "grupo": label,
                    "n": len(g),
                    "media": round(float(np.mean(g)), 4),
                    "mediana": round(float(np.median(g)), 4),
                })
        g1 = victims[resp.loc[victims.index] == "GUERRILLA"].dropna().values
        g2 = victims[resp.loc[victims.index] == "GRUPO PARAMILITAR"].dropna().values
        if len(g1) > 1 and len(g2) > 1:
            stat, p = mannwhitneyu(g1, g2, alternative="two-sided")
            effect = 1 - (2 * stat) / (len(g1) * len(g2))
            rows.append({
                "prueba": "mann_whitney_guerrilla_paramilitar",
                "variable": "presunto_responsable",
                "estadistico": round(float(stat), 4),
                "p_value": round(float(p), 6),
                "efecto_r": round(float(effect), 4),
                "decision": "diferencias" if p < 0.05 else "sin_diferencias",
            })

    if "region" in df.columns and not victims.empty:
        reg = _safe_text(df["region"])
        top_reg = reg[~reg.isin(["SIN INFORMACION"])].value_counts().head(8).index
        grupos = [victims[reg.loc[victims.index] == r].dropna().values for r in top_reg]
        grupos = [g for g in grupos if len(g) > 1]
        labels = [r for r in top_reg if len(victims[reg.loc[victims.index] == r].dropna()) > 1]
        if len(grupos) >= 2:
            stat, p = kruskal(*grupos)
            rows.append({
                "prueba": "kruskal_region",
                "variable": "region",
                "estadistico": round(float(stat), 4),
                "p_value": round(float(p), 6),
                "decision": "diferencias" if p < 0.05 else "sin_diferencias",
            })
            for label, g in zip(labels, grupos):
                group_rows.append({
                    "prueba": "kruskal_region",
                    "grupo": label,
                    "n": len(g),
                    "media": round(float(np.mean(g)), 4),
                    "mediana": round(float(np.median(g)), 4),
                })

    if "total_hechos" in df.columns and not victims.empty:
        total_hechos = pd.to_numeric(df.loc[victims.index, "total_hechos"], errors="coerce")
        mask = total_hechos.notna() & victims.notna()
        if mask.sum() > 2:
            rho, p = spearmanr(total_hechos[mask], victims[mask])
            rows.append({
                "prueba": "spearman_total_hechos_victimas",
                "variable": "total_hechos",
                "estadistico": round(float(rho), 4),
                "p_value": round(float(p), 6),
                "decision": "correlacion" if p < 0.05 else "sin_correlacion",
            })

    for col in _existing_hechos(df):
        x = pd.to_numeric(df.loc[victims.index, col], errors="coerce")
        mask = x.notna() & victims.notna()
        if mask.sum() > 2 and x[mask].nunique() > 1:
            rho, p = spearmanr(x[mask], victims[mask])
            spearman_rows.append({
                "variable": col,
                "hecho": HECHOS_LABELS.get(col, col),
                "rho": round(float(rho), 6),
                "p_value": round(float(p), 6),
                "decision": "correlacion" if p < 0.05 else "sin_correlacion",
            })

    if not victims.empty:
        mean, low, high = _bootstrap_mean(victims.values, n_resamples=1500)
        boot_rows.append({"grupo": "total", "n": len(victims), "media": round(mean, 4), "ic95_inf": round(low, 4), "ic95_sup": round(high, 4)})
        if "presunto_responsable" in df.columns:
            resp = _safe_text(df["presunto_responsable"])
            top_resp = resp[~resp.isin(["DESCONOCIDO", "SIN INFORMACION"])].value_counts().head(5).index
            for r in top_resp:
                vals = victims[resp.loc[victims.index] == r].dropna().values
                mean, low, high = _bootstrap_mean(vals, n_resamples=700)
                boot_rows.append({"grupo": r, "n": len(vals), "media": round(mean, 4), "ic95_inf": round(low, 4), "ic95_sup": round(high, 4)})

    outputs = {
        "estadistica_inferencial_avanzada": pd.DataFrame(rows),
        "estadistica_inferencial_grupos": pd.DataFrame(group_rows),
        "spearman_hechos_victimas": pd.DataFrame(spearman_rows),
        "bootstrap_ic_victimas": pd.DataFrame(boot_rows),
    }
    save_table(outputs["estadistica_inferencial_avanzada"], ADVANCED_INFERENCE_TABLE)
    save_table(outputs["estadistica_inferencial_grupos"], ADVANCED_GROUPS_TABLE)
    save_table(outputs["spearman_hechos_victimas"], SPEARMAN_TABLE)
    save_table(outputs["bootstrap_ic_victimas"], BOOTSTRAP_TABLE)
    return outputs


def plot_advanced_inference(df: pd.DataFrame, advanced_outputs: dict[str, pd.DataFrame]) -> None:
    victims = _victims(df).dropna()
    if victims.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Estadística Inferencial — Conflicto Armado Colombia", fontsize=14, fontweight="bold")

    adv = advanced_outputs.get("estadistica_inferencial_avanzada", pd.DataFrame())
    p_resp = adv.loc[adv.get("prueba", pd.Series(dtype=str)).eq("kruskal_responsable"), "p_value"] if not adv.empty else pd.Series(dtype=float)
    p_reg = adv.loc[adv.get("prueba", pd.Series(dtype=str)).eq("kruskal_region"), "p_value"] if not adv.empty else pd.Series(dtype=float)
    rho = adv.loc[adv.get("prueba", pd.Series(dtype=str)).eq("spearman_total_hechos_victimas"), "estadistico"] if not adv.empty else pd.Series(dtype=float)

    if "presunto_responsable" in df.columns:
        resp = _safe_text(df["presunto_responsable"])
        top = resp[~resp.isin(["DESCONOCIDO", "SIN INFORMACION"])].value_counts().head(5).index
        groups = [victims[resp.loc[victims.index] == r].values for r in top]
        groups = [g for g in groups if len(g) > 0]
        if groups:
            bp = axes[0, 0].boxplot(groups, patch_artist=True, flierprops=dict(marker="o", markersize=3, alpha=0.3), medianprops=dict(color="white", linewidth=2))
            for patch, color in zip(bp["boxes"], PALETTE):
                patch.set_facecolor(color); patch.set_alpha(0.7)
            axes[0, 0].set_xticks(range(1, len(groups) + 1))
            axes[0, 0].set_xticklabels([r.replace(" ", "\n") for r in top[:len(groups)]], fontsize=7)
    axes[0, 0].set_title(f"Kruskal-Wallis — Víctimas por Responsable\np={float(p_resp.iloc[0]):.4f}" if not p_resp.empty else "Kruskal-Wallis — Víctimas por Responsable", fontweight="bold")
    axes[0, 0].set_ylabel("Total de Víctimas")
    axes[0, 0].set_ylim(0, min(10, max(10, victims.quantile(0.99))))

    if "region" in df.columns:
        reg = _safe_text(df["region"])
        top = reg[reg != "SIN INFORMACION"].value_counts().head(8).index
        groups = [victims[reg.loc[victims.index] == r].values for r in top]
        groups = [g for g in groups if len(g) > 0]
        if groups:
            bp = axes[0, 1].boxplot(groups, patch_artist=True, flierprops=dict(marker="o", markersize=3, alpha=0.3), medianprops=dict(color="white", linewidth=2))
            for patch, color in zip(bp["boxes"], PALETTE * 2):
                patch.set_facecolor(color); patch.set_alpha(0.7)
            axes[0, 1].set_xticks(range(1, len(groups) + 1))
            axes[0, 1].set_xticklabels([r.replace(" ", "\n") for r in top[:len(groups)]], fontsize=6)
    axes[0, 1].set_title(f"Kruskal-Wallis — Víctimas por Región\np={float(p_reg.iloc[0]):.4f}" if not p_reg.empty else "Kruskal-Wallis — Víctimas por Región", fontweight="bold")
    axes[0, 1].set_ylabel("Total de Víctimas")
    axes[0, 1].set_ylim(0, min(10, max(10, victims.quantile(0.99))))

    if "total_hechos" in df.columns:
        x = pd.to_numeric(df.loc[victims.index, "total_hechos"], errors="coerce")
        mask = x.notna() & victims.notna()
        axes[1, 0].scatter(x[mask], victims[mask], alpha=0.25, s=10, color=RED, edgecolors="none")
        if mask.sum() > 2:
            z = np.polyfit(x[mask], victims[mask], 1)
            p_line = np.poly1d(z)
            x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
            axes[1, 0].plot(x_line, p_line(x_line), color=DARK, linewidth=2)
        label_rho = f"ρ={float(rho.iloc[0]):.3f}" if not rho.empty else ""
        axes[1, 0].set_title(f"Spearman — Hechos Simultáneos vs Víctimas {label_rho}", fontweight="bold")
        axes[1, 0].set_xlabel("Nº de Hechos Simultáneos")
        axes[1, 0].set_ylabel("Total de Víctimas")

    boot = advanced_outputs.get("bootstrap_ic_victimas", pd.DataFrame())
    boot = boot[boot.get("grupo", pd.Series(dtype=str)).ne("total")] if not boot.empty else boot
    if not boot.empty:
        y = boot["media"].astype(float).values
        low = boot["ic95_inf"].astype(float).values
        high = boot["ic95_sup"].astype(float).values
        x = np.arange(len(boot))
        axes[1, 1].errorbar(x, y, yerr=[y - low, high - y], fmt="o", color=BLUE, capsize=6, capthick=2, linewidth=2, markersize=8)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([str(g).replace(" ", "\n") for g in boot["grupo"]], fontsize=7)
    axes[1, 1].set_title("IC 95% Bootstrap — Media de Víctimas\npor Responsable", fontweight="bold")
    axes[1, 1].set_ylabel("Media de Víctimas por Caso")

    for ax in axes.ravel():
        _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, FIG_INFERENTIAL_ADVANCED)


def build_statistics_reports(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    descriptive = descriptive_statistics(df)
    numeric = numeric_summary(df)
    chi2 = chi_square_tests(df)
    advanced = advanced_inference(df)

    save_table(descriptive, DESCRIPTIVE_TABLE)
    if not numeric.empty:
        save_table(numeric, DESCRIPTIVE_TABLE.with_name("resumen_numerico.csv"))
    save_table(chi2, CHI2_TABLE)
    plot_advanced_inference(df, advanced)

    return {
        "estadistica_descriptiva": descriptive,
        "resumen_numerico": numeric,
        "pruebas_chi_cuadrado": chi2,
        **advanced,
    }
