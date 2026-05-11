import os
from typing import Any

import pandas as pd

DEFAULT_POSTGRES_URI = "postgresql+psycopg2://bdtools:bdtools@localhost:5432/bdtools"
DEFAULT_MONGO_URI = "mongodb://localhost:27017"


def load_to_postgres(df: pd.DataFrame, table_name: str = "sievcac_limpio", uri: str | None = None, if_exists: str = "replace") -> dict[str, Any]:
    uri = uri or os.getenv("POSTGRES_URI", DEFAULT_POSTGRES_URI)
    try:
        from sqlalchemy import create_engine

        engine = create_engine(uri)
        df.to_sql(table_name, engine, if_exists=if_exists, index=False)
        return {"ok": True, "backend": "postgres", "table": table_name, "rows": int(len(df))}
    except Exception as exc:
        return {"ok": False, "backend": "postgres", "error": str(exc)}


def load_geo_to_postgis(gdf, table_name: str = "sievcac_geo", uri: str | None = None, if_exists: str = "replace") -> dict[str, Any]:
    uri = uri or os.getenv("POSTGRES_URI", DEFAULT_POSTGRES_URI)
    try:
        from sqlalchemy import create_engine

        engine = create_engine(uri)
        gdf.to_postgis(table_name, engine, if_exists=if_exists, index=False)
        return {"ok": True, "backend": "postgis", "table": table_name, "rows": int(len(gdf))}
    except Exception as exc:
        return {"ok": False, "backend": "postgis", "error": str(exc)}


def load_to_mongodb(df: pd.DataFrame, collection_name: str = "sievcac_limpio", uri: str | None = None) -> dict[str, Any]:
    uri = uri or os.getenv("MONGO_URI", DEFAULT_MONGO_URI)
    try:
        from pymongo import MongoClient

        client = MongoClient(uri)
        db = client[os.getenv("MONGO_DB", "bdtools")]
        collection = db[collection_name]
        collection.delete_many({})
        records = df.where(pd.notnull(df), None).to_dict(orient="records")
        if records:
            collection.insert_many(records)
        client.close()
        return {"ok": True, "backend": "mongodb", "collection": collection_name, "rows": int(len(records))}
    except Exception as exc:
        return {"ok": False, "backend": "mongodb", "error": str(exc)}


def load_databases(df: pd.DataFrame, gdf=None) -> list[dict[str, Any]]:
    results = [load_to_postgres(df), load_to_mongodb(df)]
    if gdf is not None and len(gdf) > 0:
        results.append(load_geo_to_postgis(gdf))
    return results
