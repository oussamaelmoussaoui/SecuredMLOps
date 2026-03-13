# ─────────────────────────────────────────
#  Model/src/data/balance.py
#  Étape 4 — Rééquilibrage des classes (SMOTE)
# ─────────────────────────────────────────
#
#  COMMENT UTILISER :
#  Importé automatiquement par train.py
#  Ou directement :
#  > python Model/src/data/balance.py
#
#  INPUT  : Model/data/processed/X_train.npy
#           Model/data/processed/y_train.npy
#  OUTPUT : Model/data/processed/X_train_balanced.npy
#           Model/data/processed/y_train_balanced.npy
# ─────────────────────────────────────────

import logging
import warnings
from pathlib import Path

import numpy as np
import yaml
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "Model" / "data" / "processed"
PARAMS_FILE   = PROJECT_ROOT / "params.yaml"


def apply_balancing(X_train: np.ndarray, y_train: np.ndarray,
                     strategy: str = "smote"):
    """
    Rééquilibrer les classes du dataset d'entraînement.

    Paramètres
    ----------
    strategy : str
        'smote'        → Créer des exemples synthétiques (recommandé)
        'undersampling' → Réduire la classe majoritaire
        'combined'     → SMOTE + undersampling combinés

    Retourne
    --------
    X_resampled, y_resampled : arrays rééquilibrés
    """
    logger.info(f"\n── Rééquilibrage avec stratégie : {strategy} ──")

    # Avant
    unique, counts = np.unique(y_train, return_counts=True)
    logger.info("Avant rééquilibrage :")
    for cls, cnt in zip(unique, counts):
        label = "BENIGN" if cls == 0 else "ATTACK"
        logger.info(f"  Classe {cls} ({label}) : {cnt:,} ({cnt/len(y_train):.1%})")

    # ── Stratégies ──
    if strategy == "smote":
        # Créer des exemples synthétiques de la classe minoritaire (attaques)
        sampler = SMOTE(
            random_state=42,
            k_neighbors=5,
            sampling_strategy="auto"    # équilibre toutes les classes
        )
        X_res, y_res = sampler.fit_resample(X_train, y_train)

    elif strategy == "undersampling":
        # Réduire la classe majoritaire (trafic normal)
        sampler = RandomUnderSampler(
            random_state=42,
            sampling_strategy="auto"
        )
        X_res, y_res = sampler.fit_resample(X_train, y_train)

    elif strategy == "combined":
        # D'abord oversample la minorité, puis undersample la majorité
        over  = SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)
        under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
        pipeline = ImbPipeline([("over", over), ("under", under)])
        X_res, y_res = pipeline.fit_resample(X_train, y_train)

    else:
        raise ValueError(f"Stratégie inconnue : {strategy}. Choisir parmi: smote | undersampling | combined")

    # Après
    unique_res, counts_res = np.unique(y_res, return_counts=True)
    logger.info("\nAprès rééquilibrage :")
    for cls, cnt in zip(unique_res, counts_res):
        label = "BENIGN" if cls == 0 else "ATTACK"
        logger.info(f"  Classe {cls} ({label}) : {cnt:,} ({cnt/len(y_res):.1%})")

    logger.info(f"\nTaille dataset : {len(y_train):,} → {len(y_res):,}")
    return X_res, y_res


def main():
    """Exécution standalone du rééquilibrage."""
    logger.info("═══════════════════════════════════════")
    logger.info("  RÉÉQUILIBRAGE DES CLASSES")
    logger.info("═══════════════════════════════════════")

    # Charger params
    with open(PARAMS_FILE) as f:
        params = yaml.safe_load(f)
    strategy = params["balancing"]["strategy"]

    # Charger données
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    logger.info(f"Données chargées : X={X_train.shape} | y={y_train.shape}")

    # Rééquilibrage
    X_res, y_res = apply_balancing(X_train, y_train, strategy)

    # Sauvegarder
    np.save(PROCESSED_DIR / "X_train_balanced.npy", X_res)
    np.save(PROCESSED_DIR / "y_train_balanced.npy", y_res)
    logger.info(f"\n✅ Données sauvegardées dans {PROCESSED_DIR}")
    logger.info("   Prochaine étape :")
    logger.info("   > python Model/src/models/train.py")


if __name__ == "__main__":
    main()
