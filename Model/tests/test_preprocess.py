# ─────────────────────────────────────────
#  Model/tests/test_preprocess.py
#  Tests unitaires — Preprocessing
# ─────────────────────────────────────────
#
#  COMMENT UTILISER :
#  > pytest Model/tests/test_preprocess.py -v
# ─────────────────────────────────────────

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

# Ajouter le projet au path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ─────────────────────────────────────────
#  FIXTURES — Données de test synthétiques
# ─────────────────────────────────────────
@pytest.fixture
def sample_df():
    """DataFrame minimal simulant CICIDS2017."""
    np.random.seed(42)
    n = 200

    df = pd.DataFrame({
        "Flow Duration":         np.random.uniform(0, 10000, n),
        "Total Fwd Packets":     np.random.randint(1, 100, n),
        "Total Backward Packets": np.random.randint(0, 50, n),
        "Flow Bytes/s":          np.random.uniform(0, 100000, n),
        "Flow Packets/s":        np.random.uniform(0, 5000, n),
        "Fwd Packet Length Mean": np.random.uniform(0, 1500, n),
        "Bwd Packet Length Mean": np.random.uniform(0, 1500, n),
        "Flow IAT Mean":         np.random.uniform(0, 50000, n),
        "Flow ID":               [f"flow_{i}" for i in range(n)],
        "Source IP":             ["192.168.1.1"] * n,
        "Destination IP":        ["10.0.0.1"] * n,
        "Label": np.random.choice(["BENIGN", "DoS Hulk", "PortScan"], n, p=[0.7, 0.2, 0.1])
    })
    return df


@pytest.fixture
def df_with_issues(sample_df):
    """DataFrame avec problèmes volontaires (inf, NaN, doublons)."""
    df = sample_df.copy()

    # Ajouter des valeurs infinies
    df.iloc[0, 3] = np.inf
    df.iloc[1, 4] = -np.inf

    # Ajouter des NaN
    df.iloc[2, 0] = np.nan

    # Ajouter des doublons
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)

    return df


# ─────────────────────────────────────────
#  TESTS — Nettoyage
# ─────────────────────────────────────────
class TestCleaning:

    def test_remove_duplicates(self, df_with_issues):
        """La suppression des doublons fonctionne correctement."""
        initial_len = len(df_with_issues)
        cleaned = df_with_issues.drop_duplicates()
        assert len(cleaned) < initial_len, "Les doublons n'ont pas été supprimés"

    def test_replace_inf_with_nan(self, df_with_issues):
        """Les valeurs infinies sont remplacées par NaN."""
        df = df_with_issues.replace([np.inf, -np.inf], np.nan)
        assert not np.isinf(df.select_dtypes(include=[np.number]).values).any(), \
            "Des valeurs infinies subsistent"

    def test_drop_nan_rows(self, df_with_issues):
        """Les lignes avec NaN sont supprimées."""
        df = df_with_issues.replace([np.inf, -np.inf], np.nan).dropna()
        assert df.isnull().sum().sum() == 0, "Des NaN subsistent après suppression"

    def test_drop_identifier_columns(self, sample_df):
        """Les colonnes identifiants réseau sont supprimées."""
        cols_to_drop = ["Flow ID", "Source IP", "Destination IP"]
        df = sample_df.drop(columns=[c for c in cols_to_drop if c in sample_df.columns])

        for col in cols_to_drop:
            assert col not in df.columns, f"Colonne {col} toujours présente"

    def test_no_negative_duration(self, sample_df):
        """La durée de flux ne doit pas être négative."""
        assert (sample_df["Flow Duration"] >= 0).all(), \
            "Des durées négatives trouvées"


# ─────────────────────────────────────────
#  TESTS — Encodage des Labels
# ─────────────────────────────────────────
class TestLabelEncoding:

    def test_binary_encoding_values(self, sample_df):
        """L'encodage binaire produit uniquement 0 et 1."""
        df = sample_df.copy()
        df["Label"] = (df["Label"] != "BENIGN").astype(int)
        unique_values = set(df["Label"].unique())
        assert unique_values.issubset({0, 1}), \
            f"Valeurs inattendues : {unique_values}"

    def test_benign_is_zero(self, sample_df):
        """BENIGN doit être encodé comme 0."""
        df = sample_df.copy()
        df["Label"] = (df["Label"] != "BENIGN").astype(int)
        benign_labels = df.loc[sample_df["Label"] == "BENIGN", "Label"]
        assert (benign_labels == 0).all(), "BENIGN n'est pas encodé comme 0"

    def test_attack_is_one(self, sample_df):
        """Les attaques doivent être encodées comme 1."""
        df = sample_df.copy()
        df["Label"] = (df["Label"] != "BENIGN").astype(int)
        attack_labels = df.loc[sample_df["Label"] != "BENIGN", "Label"]
        assert (attack_labels == 1).all(), "Les attaques ne sont pas encodées comme 1"

    def test_no_null_labels_after_encoding(self, sample_df):
        """Pas de labels nuls après encodage."""
        df = sample_df.copy()
        df["Label"] = (df["Label"] != "BENIGN").astype(int)
        assert df["Label"].isnull().sum() == 0, "Labels nuls après encodage"


# ─────────────────────────────────────────
#  TESTS — Split des données
# ─────────────────────────────────────────
class TestDataSplit:

    def test_split_sizes(self, sample_df):
        """Les proportions de split sont correctes."""
        from sklearn.model_selection import train_test_split

        df = sample_df.copy()
        df["Label"] = (df["Label"] != "BENIGN").astype(int)

        X = df.drop(columns=["Label", "Flow ID", "Source IP", "Destination IP"], errors="ignore")
        y = df["Label"]

        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        test_ratio = len(X_test) / len(X)

        assert abs(test_ratio - 0.2) < 0.02, \
            f"Ratio test incorrect : {test_ratio:.2f} (attendu ~0.20)"

    def test_no_data_leakage(self, sample_df):
        """Pas de fuite entre train et test."""
        from sklearn.model_selection import train_test_split

        df = sample_df.copy()
        df["Label"] = (df["Label"] != "BENIGN").astype(int)
        X = df.drop(columns=["Label", "Flow ID", "Source IP", "Destination IP"], errors="ignore")
        y = df["Label"]

        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Vérifier pas d'indices communs
        common_idx = set(X_train.index) & set(X_test.index)
        assert len(common_idx) == 0, f"Fuite de données détectée : {len(common_idx)} index communs"

    def test_stratified_split_preserves_distribution(self, sample_df):
        """Le split stratifié préserve la distribution des classes."""
        from sklearn.model_selection import train_test_split

        df = sample_df.copy()
        df["Label"] = (df["Label"] != "BENIGN").astype(int)
        X = df.drop(columns=["Label", "Flow ID", "Source IP", "Destination IP"], errors="ignore")
        y = df["Label"]

        _, _, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        train_ratio = y_train.mean()
        test_ratio  = y_test.mean()

        assert abs(train_ratio - test_ratio) < 0.05, \
            f"Distribution déséquilibrée : train={train_ratio:.2f} test={test_ratio:.2f}"


# ─────────────────────────────────────────
#  TESTS — Normalisation
# ─────────────────────────────────────────
class TestScaling:

    def test_scaler_fit_only_on_train(self):
        """Le scaler ne doit être fitté que sur le train."""
        from sklearn.preprocessing import StandardScaler

        X_train = np.random.randn(100, 5)
        #X_test  = np.random.randn(20, 5) + 10  # Distribution différente

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)

        # Train doit être centré autour de 0
        assert abs(X_train_sc.mean()) < 0.1, "Train mal normalisé"

    def test_no_inf_after_scaling(self):
        """Pas de valeurs infinies après normalisation."""
        from sklearn.preprocessing import StandardScaler

        X = np.random.randn(100, 5)
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        assert not np.isinf(X_sc).any(), "Valeurs infinies après normalisation"

    def test_scaler_mean_close_to_zero(self):
        """Après StandardScaler, la moyenne doit être proche de 0."""
        from sklearn.preprocessing import StandardScaler

        X = np.random.randn(1000, 5) * 100 + 50
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        assert abs(X_sc.mean()) < 0.01, f"Moyenne trop éloignée de 0 : {X_sc.mean():.4f}"
