# Proyecto Final - Big Data Tools

## Análisis del dataset SIEVCAC con metodología CRISP-DM

Este proyecto corresponde al trabajo final de la asignatura Big Data Tools. El objetivo principal fue construir una solución reproducible en Python para analizar casos de reclutamiento y utilización de niños, niñas y adolescentes en el marco del conflicto armado colombiano, usando como base el dataset SIEVCAC.

El trabajo se organizó siguiendo la metodología CRISP-DM, con el fin de separar mejor las etapas del análisis y evitar que todo quedara concentrado en un solo notebook. Además, se modularizó la lógica principal en la carpeta src/bdtools, dejando el notebook como una guía de ejecución y presentación de resultados.

---

## Objetivo del proyecto

Desarrollar un flujo de análisis de datos que permita:

- Cargar y limpiar el dataset original.
- Generar estadísticas descriptivas sobre los casos registrados.
- Aplicar pruebas inferenciales básicas para identificar asociaciones entre variables categóricas.
- Procesar coordenadas geográficas y exportar datos espaciales.
- Entrenar dos modelos de clasificación supervisada.
- Orquestar el flujo completo con Prefect.
- Dejar una estructura compatible con PostgreSQL/PostGIS, MongoDB y Docker.

---

## Metodología usada

El desarrollo se organizó con base en CRISP-DM:

1. **Comprensión del negocio**  
   Se definió el problema de análisis, el contexto del dataset y el propósito técnico del proyecto.

2. **Comprensión de los datos**  
   Se revisaron dimensiones, tipos de datos, valores faltantes, duplicados y variables principales.

3. **Preparación de los datos**  
   Se limpiaron nombres de columnas, fechas, textos, coordenadas y valores nulos. También se creó la variable objetivo alto_impacto.

4. **Modelado**  
   Se entrenaron únicamente dos modelos:
   - Regresión Logística.
   - Random Forest.

5. **Evaluación**  
   Se compararon los modelos mediante métricas de clasificación y matrices de confusión.

6. **Despliegue técnico**  
   Se dejó el flujo automatizado con Prefect y una configuración de servicios con Docker Compose para PostgreSQL/PostGIS y MongoDB.

---

## Estructura del proyecto

```text
ProyectoFinalBDTools_CRISPDM/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── flows/
│   └── pipeline.py
│
├── notebooks/
│   └── Proyecto_Final.ipynb
│
├── reports/
│   ├── figures/
│   └── tables/
│
├── scripts/
│   └── check_setup.py
│
├── src/
│   └── bdtools/
│       ├── __init__.py
│       ├── config.py
│       ├── io.py
│       ├── cleaning.py
│       ├── eda.py
│       ├── statistics.py
│       ├── geo.py
│       ├── modeling.py
│       ├── databases.py
│       └── utils.py
│
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Descripción de los módulos

### `config.py`

Contiene las rutas principales del proyecto, nombres de archivos de salida, variables objetivo, columnas usadas en EDA y configuración general.

### `io.py`

Se encarga de leer el dataset original desde `data/raw`, ejecutar el proceso de limpieza y exportar los datos procesados en CSV y Parquet.

### `cleaning.py`

Incluye las funciones de preparación de datos:

- Normalización de nombres de columnas.
- Limpieza de variables textuales.
- Corrección de fechas.
- Tratamiento de valores nulos.
- Extracción de latitud y longitud.
- Creación de la variable `alto_impacto`.

### `eda.py`

Genera tablas de frecuencia, resumen general y gráficos principales del análisis exploratorio.

### `statistics.py`

Contiene la parte de estadística descriptiva e inferencial:

- Frecuencias.
- Proporciones.
- Medidas resumen.
- Intervalos de confianza.
- Pruebas chi-cuadrado entre variables categóricas y `alto_impacto`.

### `geo.py`

Construye un `GeoDataFrame` a partir de las coordenadas y exporta los datos geográficos en formato GeoJSON.

### `modeling.py`

Entrena y evalúa los dos modelos definidos para el proyecto:

- Regresión Logística.
- Random Forest.

El preprocesamiento se realiza con pipelines, separando variables numéricas y categóricas.

### `databases.py`

Incluye funciones para cargar datos limpios hacia PostgreSQL/PostGIS y MongoDB.

### `pipeline.py`

Orquesta el flujo completo con Prefect:

1. ETL.
2. EDA.
3. Estadística.
4. Geoprocesamiento.
5. Modelado.
6. Carga opcional a bases de datos.

---

## Requisitos

El proyecto fue probado con:

```text
Python 3.11
Kernel: proyectofinalbdtools
```

Librerías principales:

```text
pandas
numpy
matplotlib
scipy
scikit-learn
geopandas
prefect
sqlalchemy
psycopg2-binary
pymongo
pyarrow
openpyxl
```

## Preparación inicial

Antes de ejecutar el proyecto, ubicar el dataset original en:

```text
data/raw/Caso_Conflicto_Armado.xlsx
```

Si el archivo tiene otro nombre, el sistema intenta detectar automáticamente el primer archivo `.xlsx`, `.xls` o `.csv` disponible dentro de `data/raw`.

---

El notebook está dividido en:

1. Configuración del proyecto.
2. Comprensión del negocio.
3. Comprensión de los datos.
4. Preparación de los datos.
5. EDA y estadística descriptiva.
6. Estadística inferencial.
7. Procesamiento geoespacial.
8. Modelado.
9. Evaluación de modelos.
10. Despliegue técnico.
11. Conclusiones técnicas.

---

## Ejecución del pipeline con Prefect

Desde la raíz del proyecto:

```bash
conda activate proyectofinalbdtools
python flows/pipeline.py
```

Si la ejecución termina correctamente, debe aparecer un estado similar a:

```text
Flow run ... Finished in state Completed()
```

---

## Ejecución con Docker

Para levantar PostgreSQL/PostGIS y MongoDB:

```bash
docker compose up -d postgres mongo
```

Para verificar contenedores:

```bash
docker ps --filter "name=sievcac"
```

Para detener los servicios:

```bash
docker compose down
```

Si aparece conflicto por nombres de contenedores:

```bash
docker rm -f sievcac_postgres sievcac_mongo
docker compose up -d postgres mongo
```

---

## Carga a bases de datos

Con Docker activo, se puede ejecutar el pipeline incluyendo carga a bases:

```bash
python -c "from flows.pipeline import run_pipeline; run_pipeline(load_db=True)"
```

Esto carga los datos procesados en:

- PostgreSQL/PostGIS.
- MongoDB.

---

## Salidas generadas

### Datos procesados

```text
data/processed/sievcac_limpio.csv
data/processed/sievcac_limpio.parquet
data/processed/sievcac_geo.geojson
```

### Figuras

```text
reports/figures/casos_por_anio.png
reports/figures/top_departamentos.png
reports/figures/modalidad_casos.png
reports/figures/presunto_responsable.png
reports/figures/distribucion_geografica.png
reports/figures/matriz_confusion_logistica.png
reports/figures/matriz_confusion_random_forest.png
```

### Tablas

```text
reports/tables/calidad_datos.csv
reports/tables/resumen_general.csv
reports/tables/estadistica_descriptiva.csv
reports/tables/pruebas_chi_cuadrado.csv
reports/tables/model_metrics.csv
reports/tables/classification_report_logistica.json
reports/tables/classification_report_random_forest.json
```

---

## Variable objetivo

La variable alto_impacto se construyó a partir del número total de víctimas del caso.

```text
alto_impacto = 1 si el caso registra más de una víctima
alto_impacto = 0 si el caso registra una víctima
```

Para evitar fuga de información, la variable usada para construir la etiqueta no se utiliza como predictor directo en el entrenamiento.

---

## Modelos usados

Se entrenaron solamente dos modelos, de acuerdo con el alcance definido:

### Regresión Logística

Se usó como modelo base por su interpretación sencilla y porque permite comparar de forma clara el comportamiento frente a un modelo más flexible.

### Random Forest

Se usó como segundo modelo por su capacidad para trabajar con relaciones no lineales y variables de diferente tipo.

Ambos modelos usan el mismo conjunto de datos y el mismo esquema de preprocesamiento, para que la comparación sea más justa.

---

## Consideraciones del análisis

- El dataset contiene variables sensibles y relacionadas con hechos del conflicto armado, por lo que los resultados deben interpretarse con cuidado.
- El modelo predictivo se usa como ejercicio técnico, no como herramienta para tomar decisiones sobre casos reales.
- Algunas variables pueden tener alta cantidad de valores faltantes, por lo que se aplicaron reglas de limpieza y validación.
- La calidad del análisis depende directamente de la calidad y cobertura del dataset original.
- El componente geográfico solo usa registros con coordenadas válidas.

---

## Validación rápida del proyecto

Se puede ejecutar:

```bash
python scripts/check_setup.py
```

Este script revisa que existan carpetas principales, archivos clave y que el paquete `bdtools` pueda importarse correctamente.

---
