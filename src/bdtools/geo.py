import json

import matplotlib.pyplot as plt
import pandas as pd

from bdtools.config import FIG_GEO, GEOJSON_FILE, LAT_COL, LON_COL


def build_geodataframe(df: pd.DataFrame):
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("GeoPandas no está instalado en el kernel activo.") from exc

    if not {LAT_COL, LON_COL}.issubset(df.columns):
        return gpd.GeoDataFrame(df.copy(), geometry=[], crs="EPSG:4326")

    valid = df.dropna(subset=[LAT_COL, LON_COL]).copy()
    if valid.empty:
        return gpd.GeoDataFrame(valid, geometry=[], crs="EPSG:4326")

    geometry = gpd.points_from_xy(valid[LON_COL], valid[LAT_COL])
    return gpd.GeoDataFrame(valid, geometry=geometry, crs="EPSG:4326")


def export_geojson(gdf, path=GEOJSON_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(gdf) == 0:
        path.write_text(json.dumps({"type": "FeatureCollection", "features": []}, ensure_ascii=False), encoding="utf-8")
        return
    gdf.to_file(path, driver="GeoJSON")


def plot_geographic_distribution(gdf, path=FIG_GEO) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    if len(gdf) > 0:
        gdf.plot(ax=ax, markersize=4, alpha=0.45)
    ax.set_title("Distribución geográfica de casos")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_geospatial_outputs(df: pd.DataFrame):
    gdf = build_geodataframe(df)
    export_geojson(gdf)
    plot_geographic_distribution(gdf)
    return gdf
