from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bdtools.config import (
    FIG_CM_LOGISTIC,
    FIG_CM_RF,
    MODEL_FEATURES,
    MODEL_METRICS_TABLE,
    RANDOM_STATE,
    REPORT_LOGISTIC_JSON,
    REPORT_RF_JSON,
    TARGET_COL,
    TEST_SIZE,
)
from bdtools.utils import save_json, save_table


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def select_model_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"No existe la variable objetivo: {TARGET_COL}")

    features = [col for col in MODEL_FEATURES if col in df.columns and col != TARGET_COL]
    if not features:
        raise ValueError("No hay variables predictoras disponibles para modelar.")

    data = df[features + [TARGET_COL]].dropna(subset=[TARGET_COL]).copy()
    data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="coerce")
    data = data.dropna(subset=[TARGET_COL])
    data[TARGET_COL] = data[TARGET_COL].astype(int)

    features = [col for col in features if data[col].notna().any()]
    if not features:
        raise ValueError("No hay variables predictoras con valores observados para modelar.")

    X = data[features]
    y = data[TARGET_COL]
    numeric_features = X.select_dtypes(include=["number", "Int64", "Float64"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]
    return X, y, numeric_features, categorical_features


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="SIN INFORMACION")),
                        ("encoder", _one_hot_encoder()),
                    ]
                ),
                categorical_features,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "logistica": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def split_data(X: pd.DataFrame, y: pd.Series):
    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify)


def evaluate_classifier(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, dict, pd.DataFrame]:
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    if hasattr(model, "predict_proba") and y_test.nunique() == 2:
        y_score = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_score)
    else:
        metrics["roc_auc"] = None
    matrix = pd.DataFrame(confusion_matrix(y_test, y_pred), index=["real_0", "real_1"], columns=["pred_0", "pred_1"])
    return metrics, report, matrix


def plot_confusion_matrix(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, colorbar=False)
    ax.set_title(path.stem.replace("_", " ").title())
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate_models(df: pd.DataFrame) -> dict:
    X, y, numeric_features, categorical_features = select_model_data(df)
    if y.nunique() < 2:
        raise ValueError("La variable objetivo tiene una sola clase. No es posible entrenar modelos de clasificación.")

    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = build_models(preprocessor)

    metrics_rows = []
    reports = {}
    matrices = {}
    fitted = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics, report, matrix = evaluate_classifier(model, X_test, y_test)
        metrics_rows.append({"modelo": name, **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}})
        reports[name] = report
        matrices[name] = matrix
        fitted[name] = model

    metrics_df = pd.DataFrame(metrics_rows).sort_values("f1", ascending=False)
    save_table(metrics_df, MODEL_METRICS_TABLE)
    save_json(reports["logistica"], REPORT_LOGISTIC_JSON)
    save_json(reports["random_forest"], REPORT_RF_JSON)
    plot_confusion_matrix(fitted["logistica"], X_test, y_test, FIG_CM_LOGISTIC)
    plot_confusion_matrix(fitted["random_forest"], X_test, y_test, FIG_CM_RF)

    return {
        "models": fitted,
        "metrics": metrics_df,
        "reports": reports,
        "confusion_matrices": matrices,
        "features": {"numeric": numeric_features, "categorical": categorical_features},
        "test_data": {"X_test": X_test, "y_test": y_test},
    }


def run_modeling(df: pd.DataFrame) -> dict:
    return train_and_evaluate_models(df)
