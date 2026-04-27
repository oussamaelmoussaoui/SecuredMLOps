# ─────────────────────────────────────────
#  Model/src/models/explain.py
#  Étape 7 — Explicabilité SHAP
# ─────────────────────────────────────────
#
#  COMMENT UTILISER :
#  > python Model/src/models/explain.py
#
#  Génère des visualisations SHAP pour expliquer
#  pourquoi le modèle classe un flux comme attaque.
# ─────────────────────────────────────────

import logging
import warnings
from pathlib import Path

import joblib
import mlflow
import numpy as np
import shap
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "Model" / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "Model" / "models" / "saved"
DOCS_DIR      = PROJECT_ROOT / "Model" / "docs"
PARAMS_FILE   = PROJECT_ROOT / "params.yaml"
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    logger.info("═══════════════════════════════════════")
    logger.info("  EXPLICABILITÉ SHAP — IDS XGBoost")
    logger.info("═══════════════════════════════════════")

    # Charger le modèle
    optimized = MODELS_DIR / "xgboost_ids_optimized.joblib"
    base      = MODELS_DIR / "xgboost_ids.joblib"
    model = joblib.load(optimized if optimized.exists() else base)
    logger.info("Modèle chargé")

    # Charger les données
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    with open(PROCESSED_DIR / "feature_names.txt") as f:
        feature_names = f.read().splitlines()

    # Échantillon pour SHAP (calcul rapide)
    sample_size = min(2000, len(X_test))
    idx_sample  = np.random.choice(len(X_test), sample_size, replace=False)
    X_sample    = X_test[idx_sample]
    y_sample    = y_test[idx_sample]
    logger.info(f"Échantillon SHAP : {sample_size} exemples")

    # Créer l'explainer
    logger.info("Calcul des valeurs SHAP (peut prendre quelques minutes)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    logger.info("Valeurs SHAP calculées")

    # ── 1. Summary Plot (importance globale) ──
    logger.info("Génération : Summary Plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=20
    )
    plt.title("Impact Global des Features — XGBoost IDS")
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "shap_summary_bar.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 2. Beeswarm Plot (distribution des impacts) ──
    logger.info("Génération : Beeswarm Plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        show=False,
        max_display=15
    )
    plt.title("Distribution des Valeurs SHAP par Feature")
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "shap_beeswarm.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 3. Force Plot — Exemple d'une ATTAQUE ──
    logger.info("Génération : Force Plots individuels...")
    attack_idx = np.where(y_sample == 1)[0]
    if len(attack_idx) > 0:
        idx = attack_idx[0]
        shap.force_plot(
            explainer.expected_value,
            shap_values[idx],
            X_sample[idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.title("Explication d'une ATTAQUE détectée")
        plt.tight_layout()
        plt.savefig(DOCS_DIR / "shap_force_attack.png", dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("  Force plot ATTACK généré")

    # ── 4. Force Plot — Exemple de trafic NORMAL ──
    benign_idx = np.where(y_sample == 0)[0]
    if len(benign_idx) > 0:
        idx = benign_idx[0]
        shap.force_plot(
            explainer.expected_value,
            shap_values[idx],
            X_sample[idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.title("Explication d'un trafic BENIGN")
        plt.tight_layout()
        plt.savefig(DOCS_DIR / "shap_force_benign.png", dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("  Force plot BENIGN généré")

    # ── 5. Dependence Plot — Feature la plus importante ──
    logger.info("Génération : Dependence Plot...")
    top_feature_idx = np.abs(shap_values).mean(0).argmax()
    top_feature     = feature_names[top_feature_idx]

    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        top_feature_idx,
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False
    )
    plt.title(f"Dependence Plot — Feature : {top_feature}")
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "shap_dependence.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── Rapport texte ──
    top_features = sorted(
        zip(feature_names, np.abs(shap_values).mean(0)),
        key=lambda x: x[1], reverse=True
    )[:10]

    report_path = DOCS_DIR / "shap_report.txt"
    with open(report_path, "w") as f:
        f.write("TOP 10 FEATURES — VALEURS SHAP MOYENNES\n")
        f.write("="*45 + "\n")
        for i, (feat, val) in enumerate(top_features, 1):
            f.write(f"{i:>2}. {feat:<35} {val:.6f}\n")

    logger.info("\nTop 5 features les plus importantes :")
    for feat, val in top_features[:5]:
        logger.info(f"  {feat:<35} SHAP={val:.4f}")

    # Logger dans MLflow
    with open(PARAMS_FILE) as f:
        params = yaml.safe_load(f)
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="shap_explainability"):
        for path in DOCS_DIR.glob("shap_*.png"):
            mlflow.log_artifact(str(path))
        mlflow.log_artifact(str(report_path))
        logger.info("Artifacts SHAP loggés dans MLflow")

    logger.info("\n✅ Explicabilité SHAP terminée !")
    logger.info(f"   Visualisations dans : {DOCS_DIR}")
    logger.info("   Prochaine étape :")
    logger.info("   > python Model/src/api/main.py")


if __name__ == "__main__":
    main()
