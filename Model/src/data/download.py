# ─────────────────────────────────────────
#  Model/src/data/download.py
#  Étape 1 — Téléchargement du dataset CICIDS2017
# ─────────────────────────────────────────
#
#  COMMENT UTILISER :
#  > python Model/src/data/download.py
#
#  Ce script vérifie si le dataset existe déjà,
#  sinon il affiche les instructions pour le télécharger
#  (le dataset nécessite une inscription sur le site UNB).
# ─────────────────────────────────────────

import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Chemins ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]   # racine SECUREDMLOPS/
RAW_DATA_DIR = PROJECT_ROOT / "Model" / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "Model" / "data" / "processed"

# ── Fichiers attendus après téléchargement ──
EXPECTED_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]


def create_directories():
    """Créer les dossiers data/ s'ils n'existent pas."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Dossier raw    : {RAW_DATA_DIR}")
    logger.info(f"Dossier processed : {PROCESSED_DIR}")


def check_dataset_exists() -> bool:
    """Vérifier si au moins un fichier CSV CICIDS2017 est présent."""
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    if csv_files:
        logger.info(f"Dataset trouvé : {len(csv_files)} fichier(s) CSV dans {RAW_DATA_DIR}")
        for f in csv_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            logger.info(f"  - {f.name} ({size_mb:.1f} MB)")
        return True
    return False


def print_download_instructions():
    """Afficher les instructions de téléchargement manuel."""
    print("\n" + "="*60)
    print("  DATASET CICIDS2017 — INSTRUCTIONS DE TÉLÉCHARGEMENT")
    print("="*60)
    print("""
  Le dataset CICIDS2017 nécessite une inscription gratuite.

  OPTION 1 — Site officiel UNB (recommandé) :
  ─────────────────────────────────────────
  1. Aller sur : https://www.unb.ca/cic/datasets/ids-2017.html
  2. Cliquer sur "Download Dataset"
  3. Remplir le formulaire d'inscription
  4. Télécharger les fichiers CSV (~2.8 GB au total)
  5. Placer les fichiers dans :
     Model/data/raw/

  OPTION 2 — Kaggle (version prétraitée, plus simple) :
  ──────────────────────────────────────────────────────
  1. Aller sur : https://www.kaggle.com/datasets/cicdataset/cicids2017
  2. Télécharger le dataset
  3. Extraire les CSV dans : Model/data/raw/

  FICHIERS ATTENDUS :
  ─────────────────""")
    for f in EXPECTED_FILES:
        print(f"  ✓ {f}")
    print("""
  APRÈS TÉLÉCHARGEMENT :
  ──────────────────────
  Relancez ce script pour vérifier :
  > python Model/src/data/download.py

  Puis continuez avec le preprocessing :
  > python Model/src/data/preprocess.py
""")
    print("="*60 + "\n")


def verify_csv_structure():
    """Vérifier que les CSV ont la bonne structure CICIDS2017."""
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        return False

    import pandas as pd

    # Vérifier le premier fichier trouvé
    sample_file = csv_files[0]
    logger.info(f"Vérification de la structure : {sample_file.name}")

    try:
        df_sample = pd.read_csv(sample_file, nrows=5)
        df_sample.columns = df_sample.columns.str.strip()
        cols = df_sample.columns.tolist()

        # Colonnes essentielles CICIDS2017
        required_cols = ["Flow Duration", "Total Fwd Packets", "Label"]
        missing = [c for c in required_cols if c not in cols]

        if missing:
            logger.warning(f"Colonnes manquantes : {missing}")
            logger.warning("Le fichier ne semble pas être du format CICIDS2017 standard.")
            return False

        logger.info(f"Structure valide — {len(cols)} colonnes détectées")
        logger.info(f"Classes présentes : {df_sample['Label'].unique().tolist()}")
        return True

    except Exception as e:
        logger.error(f"Erreur lors de la vérification : {e}")
        return False


def main():
    logger.info("─── Vérification du Dataset CICIDS2017 ───")
    create_directories()

    if check_dataset_exists():
        logger.info("Vérification de la structure CSV...")
        if verify_csv_structure():
            logger.info("✅ Dataset prêt ! Prochaine étape :")
            logger.info("   > python Model/src/data/preprocess.py")
        else:
            logger.warning("⚠️  Fichiers présents mais structure incorrecte.")
            logger.warning("   Vérifiez que vous avez les bons fichiers CICIDS2017.")
    else:
        logger.warning("❌ Dataset non trouvé dans Model/data/raw/")
        print_download_instructions()
        sys.exit(1)


if __name__ == "__main__":
    main()
