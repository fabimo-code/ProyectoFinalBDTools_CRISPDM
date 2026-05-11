from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

RAW_DATA_FILE = RAW_DIR / "Caso_Conflicto_Armado.xlsx"
CLEAN_CSV = PROCESSED_DIR / "sievcac_limpio.csv"
CLEAN_PARQUET = PROCESSED_DIR / "sievcac_limpio.parquet"
GEOJSON_FILE = PROCESSED_DIR / "sievcac_geo.geojson"

QUALITY_TABLE = TABLES_DIR / "calidad_datos.csv"
SUMMARY_TABLE = TABLES_DIR / "resumen_general.csv"
DESCRIPTIVE_TABLE = TABLES_DIR / "estadistica_descriptiva.csv"
CHI2_TABLE = TABLES_DIR / "pruebas_chi_cuadrado.csv"
MODEL_METRICS_TABLE = TABLES_DIR / "model_metrics.csv"
REPORT_LOGISTIC_JSON = TABLES_DIR / "classification_report_logistica.json"
REPORT_RF_JSON = TABLES_DIR / "classification_report_random_forest.json"

NULLS_SUMMARY_TABLE = TABLES_DIR / "mapa_nulos_resumen.csv"
DROPPED_COLUMNS_TABLE = TABLES_DIR / "columnas_eliminadas_nulos.csv"
IMPLICIT_DATE_NULLS_TABLE = TABLES_DIR / "nulos_implicitos_fechas.csv"
CATEGORICAL_FREQUENCIES_TABLE = TABLES_DIR / "frecuencias_categoricas.csv"
TEXT_STANDARDIZATION_TABLE = TABLES_DIR / "estandarizacion_textos_categoricos.csv"
COORDINATES_SUMMARY_TABLE = TABLES_DIR / "coordenadas_extraidas.csv"
ADVANCED_INFERENCE_TABLE = TABLES_DIR / "estadistica_inferencial_avanzada.csv"
ADVANCED_GROUPS_TABLE = TABLES_DIR / "estadistica_inferencial_grupos.csv"
SPEARMAN_TABLE = TABLES_DIR / "spearman_hechos_victimas.csv"
BOOTSTRAP_TABLE = TABLES_DIR / "bootstrap_ic_victimas.csv"
OUTLIERS_TABLE = TABLES_DIR / "outliers_victimas.csv"

FIG_CASOS_ANIO = FIGURES_DIR / "casos_por_anio.png"
FIG_TOP_DEPARTAMENTOS = FIGURES_DIR / "top_departamentos.png"
FIG_MODALIDAD = FIGURES_DIR / "modalidad_casos.png"
FIG_RESPONSABLE = FIGURES_DIR / "presunto_responsable.png"
FIG_GEO = FIGURES_DIR / "distribucion_geografica.png"
FIG_CM_LOGISTIC = FIGURES_DIR / "matriz_confusion_logistica.png"
FIG_CM_RF = FIGURES_DIR / "matriz_confusion_random_forest.png"

FIG_MAPA_NULOS = FIGURES_DIR / "mapa_nulos.png"
FIG_PORCENTAJE_NULOS = FIGURES_DIR / "porcentaje_nulos.png"
FIG_COLUMNAS_ELIMINADAS_NULOS = FIGURES_DIR / "columnas_eliminadas_nulos.png"
FIG_DISTRIBUCION_DECADA = FIGURES_DIR / "distribucion_decada.png"
FIG_FRECUENCIAS_CATEGORICAS = FIGURES_DIR / "frecuencias_categoricas_clave.png"
FIG_MODALIDAD_FRECUENCIA_PCT = FIGURES_DIR / "modalidad_frecuencia_porcentaje.png"
FIG_TIPO_VINCULACION = FIGURES_DIR / "tipo_vinculacion.png"
FIG_FORMA_VINCULACION = FIGURES_DIR / "forma_vinculacion.png"
FIG_DEPARTAMENTOS = FIGURES_DIR / "departamentos.png"
FIG_REGIONES = FIGURES_DIR / "regiones.png"
FIG_COORDENADAS_EXTRAIDAS = FIGURES_DIR / "coordenadas_extraidas.png"

FIG_DESCRIPTIVE_PANEL = FIGURES_DIR / "estadisticas_descriptivas_conflicto.png"
FIG_VICTIMS_DISTRIBUTION = FIGURES_DIR / "distribucion_victimas_caso.png"
FIG_HECHOS_DISTRIBUTION = FIGURES_DIR / "distribucion_hechos_victimizantes.png"
FIG_RESPONSABLE_DETAIL = FIGURES_DIR / "presunto_responsable_detalle.png"
FIG_MODALIDAD_DETAIL = FIGURES_DIR / "modalidad_vinculacion_detalle.png"
FIG_TEMPORAL_DECADE_MONTH = FIGURES_DIR / "distribucion_temporal_decada_mes.png"
FIG_CONFLICT_2000_2010 = FIGURES_DIR / "conflicto_2000_2010_temporal.png"
FIG_GEO_RESP_2000_2010 = FIGURES_DIR / "geografia_responsables_2000_2010.png"
FIG_CORRELACIONES_HECHOS = FIGURES_DIR / "correlaciones_hechos.png"
FIG_RESP_MODALIDAD = FIGURES_DIR / "responsable_vs_modalidad.png"
FIG_RESP_DEPARTAMENTOS = FIGURES_DIR / "responsable_vs_departamentos.png"
FIG_VICTIMAS_BIVARIADO = FIGURES_DIR / "victimas_bivariado.png"
FIG_TEMPORAL_EVOLUCION = FIGURES_DIR / "temporal_evolucion_conflicto.png"
FIG_TEMPORAL_ACTOR = FIGURES_DIR / "temporal_actor_armado.png"
FIG_EVOLUCION_HECHOS = FIGURES_DIR / "evolucion_hechos_victimizantes.png"
FIG_GEO_CASOS_VICTIMAS = FIGURES_DIR / "geoespacial_casos_victimas.png"
FIG_OUTLIERS_VICTIMAS = FIGURES_DIR / "outliers_victimas.png"
FIG_OUTLIERS_DEPARTAMENTO = FIGURES_DIR / "outliers_departamento.png"
FIG_INFERENTIAL_ADVANCED = FIGURES_DIR / "estadistica_inferencial_avanzada.png"

RANDOM_STATE = 42
NULL_THRESHOLD = 0.90
TOP_N = 15
TEST_SIZE = 0.20

ID_COL = "id_caso"
TARGET_COL = "alto_impacto"
VICTIMS_COL = "total_victimas_caso"
LAT_COL = "latitud"
LON_COL = "longitud"
GEO_SOURCE_COL = "latitud_longitud"
YEAR_COL = "anio"
MONTH_COL = "mes"
DAY_COL = "dia"

DATE_COLUMNS = {
    "year": YEAR_COL,
    "month": MONTH_COL,
    "day": DAY_COL,
}

TEXT_COLUMNS = [
    "departamento",
    "municipio",
    "region",
    "modalidad",
    "presunto_responsable",
    "tipo_vinculacion",
    "forma_vinculacion",
]

HECHOS_COLUMNS = [
    "abandono_o_despojo_forzado_de_tierras",
    "amenaza_o_intimidacion",
    "ataque_contra_mision_medica",
    "confinamiento_o_restriccion_a_la_movilidad",
    "desplazamiento_forzado",
    "extorsion",
    "lesionados_civiles",
    "pillaje",
    "tortura",
]

HECHOS_LABELS = {
    "abandono_o_despojo_forzado_de_tierras": "Abandono/Despojo",
    "amenaza_o_intimidacion": "Amenaza/Intimidación",
    "ataque_contra_mision_medica": "Ataque Misión Médica",
    "confinamiento_o_restriccion_a_la_movilidad": "Confinamiento",
    "desplazamiento_forzado": "Desplazamiento Forzado",
    "extorsion": "Extorsión",
    "lesionados_civiles": "Lesionados Civiles",
    "pillaje": "Pillaje",
    "tortura": "Tortura",
}

EDA_COLUMNS = [
    "anio_valido",
    "departamento",
    "region",
    "modalidad",
    "presunto_responsable",
]

CATEGORICAL_REPORT_COLUMNS = [
    "modalidad",
    "tipo_vinculacion",
    "forma_vinculacion",
    "departamento",
    "region",
    "presunto_responsable",
    "decada",
]

CHI2_VARIABLES = [
    "departamento",
    "region",
    "modalidad",
    "presunto_responsable",
    "tipo_vinculacion",
    "forma_vinculacion",
    "decada",
]

MODEL_FEATURES = [
    "anio_valido",
    "mes_valido",
    "departamento",
    "municipio",
    "region",
    "modalidad",
    "presunto_responsable",
    "tipo_vinculacion",
    "forma_vinculacion",
    "latitud",
    "longitud",
    "total_hechos",
    "decada",
]

COLUMN_ALIASES = {
    "id_caso": ID_COL,
    "id": ID_COL,
    "caso": ID_COL,
    "ano": YEAR_COL,
    "anio": YEAR_COL,
    "año": YEAR_COL,
    "mes": MONTH_COL,
    "dia": DAY_COL,
    "día": DAY_COL,
    "departamento": "departamento",
    "municipio": "municipio",
    "region": "region",
    "región": "region",
    "modalidad": "modalidad",
    "presunto_responsable": "presunto_responsable",
    "tipo_vinculacion": "tipo_vinculacion",
    "tipo_de_vinculacion": "tipo_vinculacion",
    "tipo_de_vinculación": "tipo_vinculacion",
    "forma_vinculacion": "forma_vinculacion",
    "forma_de_vinculacion": "forma_vinculacion",
    "forma_de_vinculación": "forma_vinculacion",
    "total_de_victimas_del_caso": VICTIMS_COL,
    "total_victimas_del_caso": VICTIMS_COL,
    "total_victimas_caso": VICTIMS_COL,
    "total_de_víctimas_del_caso": VICTIMS_COL,
    "latitud_longitud": GEO_SOURCE_COL,
    "latitud_longitud_": GEO_SOURCE_COL,
    "latitud_longitud_1": GEO_SOURCE_COL,
    "latitud": LAT_COL,
    "longitud": LON_COL,
}

DIRECTORIES = [
    RAW_DIR,
    PROCESSED_DIR,
    FIGURES_DIR,
    TABLES_DIR,
]
