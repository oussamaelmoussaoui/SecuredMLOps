# ─────────────────────────────────────────
#  Model/src/data/preprocess.py
#  Étape 2 — Nettoyage & Prétraitement CICIDS2017
# ─────────────────────────────────────────
#
#  COMMENT UTILISER :
#  > python Model/src/data/preprocess.py
#
#  INPUT  : Model/data/raw/*.csv
#  OUTPUT : Model/data/processed/X_train.npy
#           Model/data/processed/X_val.npy
#           Model/data/processed/X_test.npy
#           Model/data/processed/y_train.npy
#           Model/data/processed/y_val.npy
#           Model/data/processed/y_test.npy
#           Model/data/processed/feature_names.txt
#           Model/data/processed/scaler.joblib
# ─────────────────────────────────────────

import os
import glob
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Chemins ──────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[3]
RAW_DIR       = PROJECT_ROOT / "Model" / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "Model" / "data" / "processed"
PARAMS_FILE   = PROJECT_ROOT / "params.yaml"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_params() -> dict:
    """Charger les paramètres depuis params.yaml"""
    with open(PARAMS_FILE, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────
#  NETTOYAGE D'UN SEUL DATAFRAME
# ─────────────────────────────────────────
def clean_single_df(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Nettoyer un seul CSV — économise la RAM."""

    # Supprimer les espaces dans les noms de colonnes
    df.columns = df.columns.str.strip()

    # Supprimer les colonnes identifiants réseau
    cols_to_drop = [c for c in params["data"]["cols_to_drop"] if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Remplacer inf par NaN — sur colonnes numériques uniquement
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    # Supprimer les lignes avec NaN ou doublons
    df = df.dropna()
    df = df.drop_duplicates()

    # Encoder le label
    label_col = params["data"]["label_column"]
    if label_col in df.columns:
        df[label_col] = (df[label_col].str.strip() != "BENIGN").astype(np.int8)

    # Convertir toutes les colonnes numériques en float32 (économise 50% RAM)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    label_cols = [label_col] if label_col in num_cols else []
    feat_cols  = [c for c in num_cols if c != label_col]
    df[feat_cols] = df[feat_cols].astype(np.float32)

    return df


# ─────────────────────────────────────────
#  CHARGEMENT FICHIER PAR FICHIER
# ─────────────────────────────────────────
def load_and_clean_all(raw_dir: Path, params: dict) -> pd.DataFrame:
    """
    Charger et nettoyer les CSV par chunks pour économiser la RAM.
    """
    files = list(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"Aucun CSV dans {raw_dir}\n"
            f"Lancez d'abord : python Model/src/data/download.py"
        )

    cleaned_dfs = []
    total_rows  = 0
    CHUNK_SIZE  = 50000   # Lire 50 000 lignes à la fois

    for i, f in enumerate(files, 1):
        logger.info(f"[{i}/{len(files)}] Traitement : {f.name}")
        file_chunks = []

        for chunk in pd.read_csv(f, encoding="utf-8",
                          chunksize=CHUNK_SIZE,
                          on_bad_lines='skip',
                          engine='python'):
            chunk = clean_single_df(chunk, params)
            file_chunks.append(chunk)

        df_file = pd.concat(file_chunks, ignore_index=True)
        rows_after = len(df_file)
        logger.info(f"  → {rows_after:,} lignes | {df_file.shape[1]} colonnes")

        cleaned_dfs.append(df_file)
        total_rows += rows_after

    logger.info(f"\nFusion de {len(cleaned_dfs)} fichiers ({total_rows:,} lignes total)...")
    combined = pd.concat(cleaned_dfs, ignore_index=True)

    before = len(combined)
    combined = combined.drop_duplicates()
    logger.info(f"Doublons supprimés : {before - len(combined):,}")
    logger.info(f"Dataset final : {combined.shape[0]:,} lignes | {combined.shape[1]} colonnes")

    return combined


# ─────────────────────────────────────────
#  SPLIT TRAIN / VAL / TEST
# ─────────────────────────────────────────
def split_data(df: pd.DataFrame, params: dict):
    """Séparation stratifiée train / val / test."""
    label_col = params["data"]["label_column"]
    test_size = params["data"]["test_size"]
    val_size  = params["data"]["val_size"]
    seed      = params["data"]["random_state"]

    logger.info("\n── Étape : Split des données ──")

    X = df.drop(columns=[label_col])
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=seed, stratify=y_train
    )

    logger.info(f"Train : {X_train.shape[0]:,} | Val : {X_val.shape[0]:,} | Test : {X_test.shape[0]:,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────
#  NORMALISATION
# ─────────────────────────────────────────
def scale_features(X_train, X_val, X_test):
    """StandardScaler — fit sur train uniquement."""
    logger.info("\n── Étape : Normalisation (StandardScaler) ──")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    scaler_path = PROCESSED_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler sauvegardé : {scaler_path}")

    return X_train_sc, X_val_sc, X_test_sc, scaler


# ─────────────────────────────────────────
#  SAUVEGARDE
# ─────────────────────────────────────────
def save_processed_data(X_train, X_val, X_test,
                         y_train, y_val, y_test,
                         feature_names: list):
    """Sauvegarder les arrays numpy."""
    logger.info("\n── Étape : Sauvegarde ──")

    arrays = {
        "X_train": X_train,       "X_val": X_val,       "X_test": X_test,
        "y_train": np.array(y_train), "y_val": np.array(y_val), "y_test": np.array(y_test),
    }
    for name, arr in arrays.items():
        path = PROCESSED_DIR / f"{name}.npy"
        np.save(path, arr)
        logger.info(f"Sauvegardé : {path.name}  shape={arr.shape}")

    feat_path = PROCESSED_DIR / "feature_names.txt"
    with open(feat_path, "w") as f:
        f.write("\n".join(feature_names))
    logger.info(f"Features : {feat_path.name} ({len(feature_names)} features)")


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
def main():
    logger.info("═══════════════════════════════════════")
    logger.info("  PREPROCESSING CICIDS2017")
    logger.info("  (traitement fichier par fichier)")
    logger.info("═══════════════════════════════════════")

    params = load_params()

    # 1. Chargement + nettoyage fichier par fichier
    logger.info("\n── Étape : Chargement & Nettoyage ──")
    df = load_and_clean_all(RAW_DIR, params)

    # 2. Afficher la distribution des classes
    label_col = params["data"]["label_column"]
    n_benign  = (df[label_col] == 0).sum()
    n_attack  = (df[label_col] == 1).sum()
    logger.info(f"\nDistribution → BENIGN={n_benign:,} | ATTACK={n_attack:,}")

    # 3. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, params)
    feature_names = X_train.columns.tolist()

    # Libérer df de la mémoire
    del df

    # 4. Normalisation
    X_train_sc, X_val_sc, X_test_sc, _ = scale_features(X_train, X_val, X_test)

    # 5. Sauvegarde
    save_processed_data(X_train_sc, X_val_sc, X_test_sc,
                        y_train, y_val, y_test, feature_names)

    logger.info("\n✅ Preprocessing terminé !")
    logger.info("   Prochaine étape :")
    logger.info("   > python Model/src/data/balance.py")


if __name__ == "__main__":
    main()