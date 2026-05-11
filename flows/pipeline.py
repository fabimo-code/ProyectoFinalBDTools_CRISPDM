from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prefect import flow, task

from bdtools.databases import load_databases
from bdtools.eda import build_eda_reports
from bdtools.geo import build_geospatial_outputs
from bdtools.io import run_etl
from bdtools.modeling import train_and_evaluate_models
from bdtools.statistics import build_statistics_reports


@task
def etl_task():
    return run_etl()


@task
def eda_task(df, quality=None):
    return build_eda_reports(df, quality)


@task
def statistics_task(df):
    return build_statistics_reports(df)


@task
def geo_task(df):
    return build_geospatial_outputs(df)


@task
def modeling_task(df):
    return train_and_evaluate_models(df)


@task
def database_task(df, gdf):
    return load_databases(df, gdf)


@flow(name="sievcac-crispdm-pipeline")
def run_pipeline(load_db: bool = False):
    df, quality = etl_task()
    eda = eda_task(df, quality)
    stats = statistics_task(df)
    gdf = geo_task(df)
    models = modeling_task(df)
    db_results = database_task(df, gdf) if load_db else []
    return {
        "rows": len(df),
        "quality_rows": len(quality),
        "eda_outputs": list(eda.keys()),
        "statistics_outputs": list(stats.keys()),
        "model_metrics": models["metrics"].to_dict(orient="records"),
        "database_results": db_results,
    }


if __name__ == "__main__":
    run_pipeline(load_db=False)
