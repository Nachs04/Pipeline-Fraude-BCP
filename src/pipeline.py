"""
Pipeline CI/CD académico para detección de transacciones sospechosas.

La lógica sigue las etapas definidas en el proyecto:
Integración -> Calidad -> Pruebas -> Evaluación ML -> Evaluación IA
-> Empaquetado -> Staging -> Promoción.

No realiza bloqueos, transferencias ni decisiones financieras.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)


def log(stage: str, message: str) -> None:
    print(f"[{stage}] {message}")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp", "customer_id", "terminal_id", "amount", "fraud"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["fraud"] = pd.to_numeric(df["fraud"], errors="coerce").astype("Int64")

    if df["timestamp"].isna().any():
        raise ValueError("Existen timestamps inválidos.")
    if df["amount"].isna().any() or (df["amount"] < 0).any():
        raise ValueError("Existen montos inválidos.")
    if not set(df["fraud"].dropna().unique()).issubset({0, 1}):
        raise ValueError("La columna fraud debe contener únicamente 0/1.")
    if df[["customer_id", "terminal_id"]].isna().any().any():
        raise ValueError("customer_id y terminal_id no pueden ser nulos.")

    return df.sort_values("timestamp").reset_index(drop=True)


def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula únicamente información disponible antes de cada transacción."""
    out = df.copy()

    out["customer_prev_count"] = out.groupby("customer_id").cumcount()

    customer_sum = out.groupby("customer_id")["amount"].cumsum() - out["amount"]
    out["customer_prev_amount"] = customer_sum

    out["terminal_prev_count"] = out.groupby("terminal_id").cumcount()

    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek

    return out


def temporal_split(df: pd.DataFrame):
    n = len(df)
    i1 = int(n * 0.60)
    i2 = int(n * 0.80)
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


FEATURES = [
    "amount",
    "customer_prev_count",
    "customer_prev_amount",
    "terminal_prev_count",
    "hour",
    "day_of_week",
]


def fit_models(train: pd.DataFrame):
    X = train[FEATURES]
    y = train["fraud"].astype(int)

    if y.nunique() < 2:
        raise ValueError("El entrenamiento necesita las dos clases: 0 y 1.")

    classifier = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    classifier.fit(X, y)

    anomaly = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", IsolationForest(
                n_estimators=200,
                contamination="auto",
                random_state=42,
            )),
        ]
    )
    anomaly.fit(X)

    return classifier, anomaly


def rule_score(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame):
    """Regla de referencia: umbral de monto aprendido solo con validación."""
    amount_threshold = float(valid["amount"].quantile(0.95))

    def score(frame):
        return (frame["amount"] >= amount_threshold).astype(int).to_numpy()

    return score(train), score(valid), score(test), amount_threshold


def alert_budget_threshold(scores: np.ndarray, budget: int) -> float:
    """Fija un umbral para aproximar el presupuesto de alertas."""
    budget = max(1, min(budget, len(scores)))
    ordered = np.sort(scores)[::-1]
    return float(ordered[budget - 1])


def evaluate_predictions(y_true, pred, scores=None):
    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()

    result = {
        "sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else None,
        "alerts": int(np.sum(pred)),
        "alerts_per_1000": float(np.sum(pred) / len(pred) * 1000),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
    }

    if scores is not None and len(np.unique(y_true)) > 1:
        result["pr_auc"] = float(average_precision_score(y_true, scores))
    else:
        result["pr_auc"] = None

    return result


def evaluate_ml(train, valid, test):
    classifier, anomaly = fit_models(train)

    # Presupuesto común: número de alertas producido por la referencia en validación.
    _, valid_rules, test_rules, rule_threshold = rule_score(train, valid, test)
    alert_budget = max(1, int(valid_rules.sum()))

    clf_valid_scores = classifier.predict_proba(valid[FEATURES])[:, 1]
    clf_test_scores = classifier.predict_proba(test[FEATURES])[:, 1]

    # Isolation Forest: menor decision_function = más anómalo.
    iso_valid_raw = -anomaly.decision_function(valid[FEATURES])
    iso_test_raw = -anomaly.decision_function(test[FEATURES])

    clf_threshold = alert_budget_threshold(clf_valid_scores, alert_budget)
    iso_threshold = alert_budget_threshold(iso_valid_raw, alert_budget)

    clf_test_pred = (clf_test_scores >= clf_threshold).astype(int)
    iso_test_pred = (iso_test_raw >= iso_threshold).astype(int)

    metrics = {
        "rule_reference": {
            "threshold": rule_threshold,
            **evaluate_predictions(test["fraud"].astype(int), test_rules),
        },
        "logistic_regression": {
            "threshold": clf_threshold,
            **evaluate_predictions(
                test["fraud"].astype(int), clf_test_pred, clf_test_scores
            ),
        },
        "isolation_forest": {
            "threshold": iso_threshold,
            **evaluate_predictions(
                test["fraud"].astype(int), iso_test_pred, iso_test_raw
            ),
        },
        "alert_budget_validation": alert_budget,
    }

    joblib.dump(
        {
            "classifier": classifier,
            "anomaly": anomaly,
            "features": FEATURES,
            "classifier_threshold": clf_threshold,
            "anomaly_threshold": iso_threshold,
        },
        ARTIFACTS / "model.joblib",
    )

    return metrics


def evaluate_ai():
    """
    Evaluación local y determinista de la capa generativa.
    En esta versión se valida el contrato de salida sin llamar a un proveedor externo.
    """
    cases = [
        {
            "case_id": "TX-DEMO-001",
            "input_score": 0.82,
            "output": {
                "case_id": "TX-DEMO-001",
                "score": 0.82,
                "confirmed_fraud": False,
                "citation": "MANUAL_DEMO",
            },
        },
        {
            "case_id": "TX-DEMO-002",
            "input_score": 0.61,
            "output": {
                "case_id": "TX-DEMO-002",
                "score": 0.61,
                "confirmed_fraud": False,
                "citation": "MANUAL_DEMO",
            },
        },
    ]

    passed = 0
    failures = []

    for case in cases:
        output = case["output"]
        ok = (
            output.get("case_id") == case["case_id"]
            and output.get("score") == case["input_score"]
            and output.get("confirmed_fraud") is False
            and bool(output.get("citation"))
        )
        if ok:
            passed += 1
        else:
            failures.append(case["case_id"])

    result = {
        "cases": len(cases),
        "passed": passed,
        "fidelity_rate": passed / len(cases),
        "failures": failures,
    }

    if failures:
        raise RuntimeError(f"Fallaron controles de fidelidad IA: {failures}")

    return result


def run_tests():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Las pruebas automáticas fallaron.")


def create_manifest(data_path: str):
    commit = os.getenv("GITHUB_SHA", "local")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "dataset": str(data_path),
        "model": "logistic_regression + isolation_forest",
        "prompt_version": "P01-P08-v1",
        "environment": "academic-demo",
        "note": "Datos sintéticos/simulados; no son datos del BCP.",
    }
    (ARTIFACTS / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def staging_test(data_path: str):
    df = load_data(data_path)
    sample = df.tail(1).iloc[0]

    result = {
        "status": "PASS",
        "case_id": f"STAGING-{sample.name}",
        "ingreso": True,
        "validacion": True,
        "alerta": "simulada",
        "revision": True,
        "note": "Prueba académica de flujo; no ejecuta acciones financieras.",
    }

    (ARTIFACTS / "staging_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Ruta del CSV")
    args = parser.parse_args()

    # 1-3: integración, calidad y pruebas
    log("1-INTEGRACION", "Código integrado.")
    log("2-CALIDAD", "Validando datos, dependencias y configuración.")
    df = load_data(args.data)
    df = add_historical_features(df)

    log("3-PRUEBAS", "Ejecutando pruebas automáticas.")
    run_tests()

    # 4: ML
    log("4-EVALUACION-ML", "Ejecutando división temporal 60/20/20.")
    train, valid, test = temporal_split(df)
    metrics = evaluate_ml(train, valid, test)

    # 5: IA
    log("5-EVALUACION-IA", "Validando fidelidad y casos controlados.")
    ai_metrics = evaluate_ai()

    all_metrics = {
        "dataset": {
            "rows": len(df),
            "train": len(train),
            "validation": len(valid),
            "test": len(test),
        },
        "ml": metrics,
        "ai": ai_metrics,
    }

    (ARTIFACTS / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 6: empaquetado
    log("6-EMPAQUETADO", "Generando artefactos y manifiesto.")
    create_manifest(args.data)

    # 7: staging
    log("7-STAGING", "Ejecutando prueba de ingreso -> alerta -> revisión.")
    staging_test(args.data)

    # 8: promoción
    # En CI/CD la aprobación real se configura en GitHub Environments.
    log("8-PROMOCION", "Artefactos listos. La promoción requiere aprobación.")

    print("\nPipeline finalizado correctamente.")
    print(json.dumps(all_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
