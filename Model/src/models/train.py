# ─────────────────────────────────────────
#  Model/src/models/train.py
#  Étape 5 — Entraînement du modèle XGBoost
# ─────────────────────────────────────────

import logging
import os
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import xgboost as xgb
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "Model" / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "Model" / "models" / "saved"
DOCS_DIR      = PROJECT_ROOT / "Model" / "docs"
PARAMS_FILE   = PROJECT_ROOT / "params.yaml"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def load_params() -> dict:
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def load_data():
    """Charger toutes les données preprocessées."""
    logger.info("Chargement des données...")

    balanced_path = PROCESSED_DIR / "X_train_balanced.npy"
    if balanced_path.exists():
        X_train = np.load(PROCESSED_DIR / "X_train_balanced.npy")
        y_train = np.load(PROCESSED_DIR / "y_train_balanced.npy")
        logger.info("Données rééquilibrées (SMOTE) utilisées pour le train")
    else:
        X_train = np.load(PROCESSED_DIR / "X_train.npy")
        y_train = np.load(PROCESSED_DIR / "y_train.npy")
        logger.info("Données originales utilisées pour le train")

    X_val   = np.load(PROCESSED_DIR / "X_val.npy")
    y_val   = np.load(PROCESSED_DIR / "y_val.npy")
    X_test  = np.load(PROCESSED_DIR / "X_test.npy")
    y_test  = np.load(PROCESSED_DIR / "y_test.npy")

    with open(PROCESSED_DIR / "feature_names.txt") as f:
        feature_names = f.read().splitlines()

    logger.info(f"Train : {X_train.shape} | Val : {X_val.shape} | Test : {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Calculer toutes les métriques d'évaluation."""
    # ✅ FIX : roc_auc_score adapté automatiquement binaire ou multiclasse
    n_classes = len(np.unique(y_true))
    if n_classes == 2:
        auc = float(roc_auc_score(y_true, y_proba[:, 1]))
    else:
        auc = float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted'))

    return {
        "accuracy":            float((y_pred == y_true).mean()),
        "f1_score":            float(f1_score(y_true, y_pred, average="weighted")),
        "precision":           float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall":              float(recall_score(y_true, y_pred, average="weighted")),
        "roc_auc":             auc,
        "false_positive_rate": float((y_pred[y_true == 0] == 1).mean()),
        "false_negative_rate": float((y_pred[y_true == 1] == 0).mean()),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: Path):
    """Sauvegarder la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["BENIGN", "ATTACK"],
        yticklabels=["BENIGN", "ATTACK"],
        ylabel="Vraie classe",
        xlabel="Classe prédite",
        title=title
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    logger.info(f"Matrice de confusion sauvegardée : {save_path.name}")


def train_baseline(X_train, y_train, X_val, y_val, params: dict) -> dict:
    """Modèle de référence — Logistic Regression."""
    logger.info("\n── Entraînement Baseline (Logistic Regression) ──")

    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="baseline_logistic_regression"):
        model_params = {"C": 1.0, "max_iter": 1000, "random_state": 42, "n_jobs": -1}
        mlflow.log_params(model_params)

        model = LogisticRegression(**model_params)
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_val)
        y_proba = model.predict_proba(X_val)
        metrics = compute_metrics(y_val, y_pred, y_proba)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "baseline_model")

        cm_path = DOCS_DIR / "confusion_matrix_baseline.png"
        plot_confusion_matrix(y_val, y_pred, "Baseline — Logistic Regression", cm_path)
        mlflow.log_artifact(str(cm_path))

        logger.info(f"Baseline — F1={metrics['f1_score']:.4f} | AUC={metrics['roc_auc']:.4f}")
        return metrics


def train_xgboost(X_train, y_train, X_val, y_val,
                   X_test, y_test, feature_names: list,
                   params: dict) -> tuple:
    """Modèle principal — XGBoost."""
    logger.info("\n── Entraînement XGBoost (Modèle Principal) ──")

    # ✅ FIX : utilise la variable d'env MLFLOW_EXPERIMENT_NAME si présente
    experiment_name = os.environ.get(
        "MLFLOW_EXPERIMENT_NAME",
        params["mlflow"]["experiment_name"]
    )
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="xgboost_ids_detector") as run:

        p = params["model"]
        model_params = {
            "n_estimators":       p["n_estimators"],
            "max_depth":          p["max_depth"],
            "learning_rate":      p["learning_rate"],
            "subsample":          p["subsample"],
            "colsample_bytree":   p["colsample_bytree"],
            "scale_pos_weight":   p["scale_pos_weight"],
            "random_state":       p["random_state"],
            "n_jobs":             -1,
            "eval_metric":        "logloss",
            "use_label_encoder":  False,
        }
        mlflow.log_params(model_params)

        model = xgb.XGBClassifier(
            **model_params,
            early_stopping_rounds=p["early_stopping_rounds"]
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100
        )

        # ── Évaluation sur Validation ──
        y_pred_val  = model.predict(X_val)
        y_proba_val = model.predict_proba(X_val)
        val_metrics = compute_metrics(y_val, y_pred_val, y_proba_val)

        logger.info("\n── Résultats sur Validation ──")
        for k, v in val_metrics.items():
            logger.info(f"  {k:<30} : {v:.4f}")
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        # ── Évaluation sur Test ──
        y_pred_test  = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test)
        test_metrics = compute_metrics(y_test, y_pred_test, y_proba_test)

        logger.info("\n── Résultats sur Test (Final) ──")
        for k, v in test_metrics.items():
            logger.info(f"  {k:<30} : {v:.4f}")
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # ── Rapport de classification ──
        report = classification_report(
            y_test, y_pred_test,
            target_names=["BENIGN", "ATTACK"]
        )
        logger.info(f"\nRapport de classification :\n{report}")

        report_path = DOCS_DIR / "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(str(report_path))

        # ── Matrices de confusion ──
        cm_val_path  = DOCS_DIR / "confusion_matrix_val.png"
        cm_test_path = DOCS_DIR / "confusion_matrix_test.png"
        plot_confusion_matrix(y_val, y_pred_val, "XGBoost — Validation", cm_val_path)
        plot_confusion_matrix(y_test, y_pred_test, "XGBoost — Test", cm_test_path)
        mlflow.log_artifact(str(cm_val_path))
        mlflow.log_artifact(str(cm_test_path))

        # ── Importance des features ──
        importance = model.feature_importances_
        top_n = min(20, len(feature_names))
        idx   = importance.argsort()[-top_n:][::-1]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(range(top_n), importance[idx], color="steelblue")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in idx], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f"Top {top_n} Features Importantes — XGBoost IDS")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        feat_path = DOCS_DIR / "feature_importance.png"
        plt.savefig(feat_path, dpi=120)
        plt.close()
        mlflow.log_artifact(str(feat_path))

        # ── Logger le modèle dans MLflow ──
        mlflow.xgboost.log_model(
            model,
            artifact_path="xgboost_ids_model",
            registered_model_name=params["mlflow"]["model_registry_name"]
        )

        # ── Sauvegarder localement ──
        local_path = MODELS_DIR / "xgboost_ids.joblib"
        joblib.dump(model, local_path)
        logger.info(f"\nModèle sauvegardé localement : {local_path}")

        run_id = run.info.run_id
        logger.info(f"MLflow Run ID : {run_id}")
        # ✅ FIX : URL MLflow dynamique selon l'URI réelle (plus localhost hardcodé)
        tracking_uri = mlflow.get_tracking_uri()
        logger.info(f"Voir l'expérience : {tracking_uri}")

        return model, test_metrics, run_id


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
def main():
    logger.info("═══════════════════════════════════════")
    logger.info("  ENTRAÎNEMENT DU MODÈLE IDS")
    logger.info("═══════════════════════════════════════")

    params = load_params()

    # ✅ FIX PRINCIPAL : lire MLFLOW_TRACKING_URI depuis l'environnement EN PRIORITÉ
    #    Si absent (local), fallback sur params.yaml
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        params["mlflow"]["tracking_uri"]
    )
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow Tracking URI : {tracking_uri}")

    # Charger les données
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_data()

    logger.info("\n" + "="*40)
    model, test_metrics, run_id = train_xgboost(
        X_train, y_train, X_val, y_val,
        X_test, y_test, feature_names, params
    )

    logger.info("\n✅ Entraînement terminé !")
    logger.info("   Prochaine étape :")
    logger.info("   > python Model/src/models/evaluate.py")
    logger.info("   > python Model/src/models/explain.py")


if __name__ == "__main__":
    main()
