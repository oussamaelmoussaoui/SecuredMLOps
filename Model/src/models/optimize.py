# ─────────────────────────────────────────
#  Model/src/models/optimize.py
#  Étape 6 — Optimisation Hyperparamètres (Optuna)
# ─────────────────────────────────────────
#
#  COMMENT UTILISER :
#  > python Model/src/models/optimize.py
#
#  Lance 50 essais automatiques pour trouver
#  les meilleurs hyperparamètres XGBoost.
#  Les résultats sont loggés dans MLflow.
# ─────────────────────────────────────────

import logging
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import xgboost as xgb
import yaml
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "Model" / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "Model" / "models" / "saved"
PARAMS_FILE   = PROJECT_ROOT / "params.yaml"


def load_params():
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def objective(trial, X_train, y_train, X_val, y_val) -> float:
    """
    Fonction objectif Optuna.
    Chaque 'trial' teste une combinaison d'hyperparamètres.
    Retourne le F1-Score sur la validation.
    """
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 2.0),
        "random_state":     42,
        "n_jobs":           -1,
        "eval_metric":      "logloss",
        "use_label_encoder": False,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=30,
        verbose=False
    )

    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred, average="weighted")


def run_optimization(params: dict):
    """Lancer l'optimisation Optuna avec logging MLflow."""

    # Charger les données
    balanced_path = PROCESSED_DIR / "X_train_balanced.npy"
    if balanced_path.exists():
        X_train = np.load(PROCESSED_DIR / "X_train_balanced.npy")
        y_train = np.load(PROCESSED_DIR / "y_train_balanced.npy")
    else:
        X_train = np.load(PROCESSED_DIR / "X_train.npy")
        y_train = np.load(PROCESSED_DIR / "y_train.npy")

    X_val = np.load(PROCESSED_DIR / "X_val.npy")
    y_val = np.load(PROCESSED_DIR / "y_val.npy")

    n_trials = params["optuna"]["n_trials"]
    logger.info(f"Lancement de {n_trials} essais Optuna...")

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="optuna_hyperparameter_search"):

        study = optuna.create_study(
            direction="maximize",
            study_name="ids_xgboost_optimization",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        # Callback pour afficher la progression
        def callback(study, trial):
            if trial.number % 10 == 0:
                logger.info(
                    f"  Trial {trial.number:>3} | "
                    f"F1={trial.value:.4f} | "
                    f"Best={study.best_value:.4f}"
                )

        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            callbacks=[callback],
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_value  = study.best_value

        logger.info(f"\n── Meilleurs hyperparamètres trouvés ──")
        for k, v in best_params.items():
            logger.info(f"  {k:<25} : {v}")
        logger.info(f"\n  Meilleur F1-Score : {best_value:.6f}")

        # Logger dans MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score", best_value)
        mlflow.log_metric("n_trials", n_trials)

        # Sauvegarder les meilleurs params dans un fichier YAML
        best_params_path = MODELS_DIR / "best_params.yaml"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(best_params_path, "w") as f:
            yaml.dump({"best_params": best_params, "best_f1": best_value}, f)
        mlflow.log_artifact(str(best_params_path))
        logger.info(f"\nMeilleurs params sauvegardés : {best_params_path}")

        # Réentraîner avec les meilleurs params et sauvegarder
        logger.info("\nRéentraînement avec les meilleurs hyperparamètres...")
        best_model = xgb.XGBClassifier(
            **best_params,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                       early_stopping_rounds=30, verbose=False)

        joblib.dump(best_model, MODELS_DIR / "xgboost_ids_optimized.joblib")
        mlflow.xgboost.log_model(
            best_model,
            artifact_path="xgboost_optimized",
            registered_model_name=f"{params['mlflow']['model_registry_name']}_Optimized"
        )

        return best_params, best_value, study


def main():
    logger.info("═══════════════════════════════════════")
    logger.info("  OPTIMISATION HYPERPARAMÈTRES (OPTUNA)")
    logger.info("═══════════════════════════════════════")

    params = load_params()
    best_params, best_f1, study = run_optimization(params)

    logger.info(f"\n✅ Optimisation terminée ! Meilleur F1 : {best_f1:.4f}")
    logger.info("   Prochaine étape :")
    logger.info("   > python Model/src/models/explain.py")


if __name__ == "__main__":
    main()
