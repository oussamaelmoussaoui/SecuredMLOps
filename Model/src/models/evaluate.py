# ─────────────────────────────────────────
#  Model/src/models/evaluate.py
#  Évaluation complète du modèle entraîné
# ─────────────────────────────────────────
#
#  COMMENT UTILISER :
#  > python Model/src/models/evaluate.py
#
#  INPUT  : Model/models/saved/xgboost_ids.joblib
#           Model/data/processed/X_test.npy
#           Model/data/processed/y_test.npy
#  OUTPUT : Model/docs/evaluation_report.html
# ─────────────────────────────────────────

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "Model" / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "Model" / "models" / "saved"
DOCS_DIR      = PROJECT_ROOT / "Model" / "docs"
PARAMS_FILE   = PROJECT_ROOT / "params.yaml"
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def load_model_and_data():
    # Essayer le modèle optimisé d'abord
    optimized = MODELS_DIR / "xgboost_ids_optimized.joblib"
    base      = MODELS_DIR / "xgboost_ids.joblib"
    model_path = optimized if optimized.exists() else base

    model = joblib.load(model_path)
    logger.info(f"Modèle chargé : {model_path.name}")

    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    with open(PROCESSED_DIR / "feature_names.txt") as f:
        feature_names = f.read().splitlines()

    return model, X_test, y_test, feature_names


def plot_roc_curve(y_test, y_proba, save_path):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("Taux de Faux Positifs")
    plt.ylabel("Taux de Vrais Positifs (Recall)")
    plt.title("Courbe ROC — XGBoost IDS")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    return auc


def plot_precision_recall_curve(y_test, y_proba, save_path):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="darkorange", lw=2, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Courbe Precision-Recall — XGBoost IDS")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    return ap


def generate_html_report(metrics: dict, report_str: str, save_path: Path):
    """Rapport HTML complet de l'évaluation."""
    rows = ""
    for k, v in metrics.items():
        status_color = ""
        if k == "f1_score" and v > 0.97:     
            status_color = "background:#eafaf1"
        elif k == "roc_auc" and v > 0.99:    
            status_color = "background:#eafaf1"
        elif k == "false_positive_rate" and v < 0.05: 
            status_color = "background:#eafaf1"
        rows += f"<tr style='{status_color}'><td><b>{k}</b></td><td>{v:.6f}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Rapport d'Évaluation — IDS XGBoost</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; }}
        h1, h2 {{ color: #2c3e50; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #2c3e50; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        pre {{ background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        .images {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        img {{ width: 45%; border: 1px solid #ddd; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>📊 Rapport d'Évaluation — XGBoost IDS CICIDS2017</h1>

    <h2>Métriques de Performance</h2>
    <table>
        <tr><th>Métrique</th><th>Valeur</th></tr>
        {rows}
    </table>

    <h2>Rapport de Classification Détaillé</h2>
    <pre>{report_str}</pre>

    <h2>Visualisations</h2>
    <div class="images">
        <img src="confusion_matrix_test.png" alt="Matrice de confusion">
        <img src="roc_curve.png" alt="Courbe ROC">
        <img src="precision_recall_curve.png" alt="Courbe Precision-Recall">
        <img src="feature_importance.png" alt="Feature Importance">
    </div>
</body>
</html>"""

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Rapport HTML : {save_path}")


def main():
    logger.info("═══════════════════════════════════════")
    logger.info("  ÉVALUATION DU MODÈLE IDS")
    logger.info("═══════════════════════════════════════")

    model, X_test, y_test, feature_names = load_model_and_data()

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":            float((y_pred == y_test).mean()),
        "f1_score":            float(f1_score(y_test, y_pred, average="weighted")),
        "precision":           float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall":              float(recall_score(y_test, y_pred, average="weighted")),
        "roc_auc":             float(roc_auc_score(y_test, y_proba)),
        "false_positive_rate": float((y_pred[y_test == 0] == 1).mean()),
        "false_negative_rate": float((y_pred[y_test == 1] == 0).mean()),
    }

    logger.info("\n── Métriques Finales ──")
    for k, v in metrics.items():
        logger.info(f"  {k:<30} : {v:.4f}")

    # Rapport de classification
    report_str = classification_report(y_test, y_pred, target_names=["BENIGN", "ATTACK"])
    logger.info(f"\n{report_str}")

    # Courbes
    plot_roc_curve(y_test, y_proba, DOCS_DIR / "roc_curve.png")
    plot_precision_recall_curve(y_test, y_proba, DOCS_DIR / "precision_recall_curve.png")

    # Rapport HTML
    generate_html_report(metrics, report_str, DOCS_DIR / "evaluation_report.html")

    # Vérification des seuils
    logger.info("\n── Vérification des Seuils Cibles ──")
    checks = {
        "F1 > 0.97":          metrics["f1_score"] > 0.97,
        "ROC AUC > 0.99":     metrics["roc_auc"] > 0.99,
        "FPR < 5%":           metrics["false_positive_rate"] < 0.05,
        "Recall > 95%":       metrics["recall"] > 0.95,
    }
    all_ok = True
    for check, passed in checks.items():
        icon = "✅" if passed else "⚠️"
        logger.info(f"  {icon} {check}")
        if not passed:
            all_ok = False

    if all_ok:
        logger.info("\n✅ Toutes les cibles atteintes !")
        logger.info("   Prochaine étape :")
        logger.info("   > python Model/src/models/explain.py")
    else:
        logger.warning("\n⚠️  Certaines cibles non atteintes.")
        logger.warning("   Considérer : python Model/src/models/optimize.py")


if __name__ == "__main__":
    main()
