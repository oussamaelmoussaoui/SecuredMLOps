# ─────────────────────────────────────────
#  Model/tests/test_model.py
#  Tests unitaires — Modèle XGBoost
# ─────────────────────────────────────────
#
#  COMMENT UTILISER :
#  > pytest Model/tests/test_model.py -v
#
#  Pour tous les tests :
#  > pytest Model/tests/ -v --cov=Model/src
# ─────────────────────────────────────────

import time
from pathlib import Path
import sys

import numpy as np
import pytest
import xgboost as xgb
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ─────────────────────────────────────────
#  FIXTURES
# ─────────────────────────────────────────
@pytest.fixture(scope="module")
def trained_model():
    """Créer et entraîner un modèle XGBoost minimal pour les tests."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        weights=[0.8, 0.2],   # Simuler le déséquilibre CICIDS2017
        random_state=42
    )
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=4,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1
    )
    model.fit(X[:800], y[:800])
    return model, X[800:], y[800:]


@pytest.fixture
def single_benign_sample():
    """Un seul exemple de trafic normal (synthétique)."""
    np.random.seed(1)
    return np.random.randn(1, 20) * 0.5   # Petite variance = trafic normal


@pytest.fixture
def attack_samples():
    """Lot d'exemples d'attaque (synthétiques)."""
    np.random.seed(99)
    return np.random.randn(50, 20) * 3    # Grande variance = attaques


# ─────────────────────────────────────────
#  TESTS — Format des prédictions
# ─────────────────────────────────────────
class TestPredictionFormat:

    def test_predict_returns_0_or_1(self, trained_model, single_benign_sample):
        """La prédiction doit être 0 ou 1."""
        model, _, _ = trained_model
        pred = model.predict(single_benign_sample)
        assert pred[0] in [0, 1], f"Valeur inattendue : {pred[0]}"

    def test_predict_proba_shape(self, trained_model, single_benign_sample):
        """predict_proba doit retourner 2 colonnes (BENIGN, ATTACK)."""
        model, _, _ = trained_model
        proba = model.predict_proba(single_benign_sample)
        assert proba.shape == (1, 2), f"Shape incorrecte : {proba.shape}"

    def test_probabilities_sum_to_one(self, trained_model, single_benign_sample):
        """Les probabilités doivent sommer à 1."""
        model, _, _ = trained_model
        proba = model.predict_proba(single_benign_sample)
        total = proba[0].sum()
        assert abs(total - 1.0) < 1e-6, f"Somme des probas : {total:.6f} (attendu 1.0)"

    def test_probability_range(self, trained_model):
        """Toutes les probabilités doivent être entre 0 et 1."""
        model, X_test, _ = trained_model
        probas = model.predict_proba(X_test)
        assert (probas >= 0).all(), "Probabilité négative détectée"
        assert (probas <= 1).all(), "Probabilité > 1 détectée"

    def test_batch_prediction_size(self, trained_model):
        """La prédiction batch doit retourner autant de résultats que d'inputs."""
        model, X_test, _ = trained_model
        preds = model.predict(X_test)
        assert len(preds) == len(X_test), \
            f"Taille incorrecte : {len(preds)} (attendu {len(X_test)})"


# ─────────────────────────────────────────
#  TESTS — Performance
# ─────────────────────────────────────────
class TestPerformance:

    def test_inference_latency_single(self, trained_model, single_benign_sample):
        """L'inférence sur un seul exemple doit être < 100ms."""
        model, _, _ = trained_model
        start   = time.time()
        model.predict(single_benign_sample)
        elapsed = (time.time() - start) * 1000
        assert elapsed < 100, f"Trop lent : {elapsed:.2f}ms (max: 100ms)"

    def test_inference_latency_batch(self, trained_model):
        """L'inférence sur 1000 exemples doit être < 2000ms."""
        model, X_test, _ = trained_model
        X_large = np.tile(X_test, (5, 1))[:1000]   # 1000 exemples
        start   = time.time()
        model.predict(X_large)
        elapsed = (time.time() - start) * 1000
        assert elapsed < 2000, f"Trop lent pour le batch : {elapsed:.2f}ms"

    def test_model_accuracy_above_threshold(self, trained_model):
        """L'accuracy doit être > 80% sur les données de test."""
        model, X_test, y_test = trained_model
        y_pred   = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        assert accuracy > 0.80, f"Accuracy trop faible : {accuracy:.2%} (min: 80%)"

    def test_f1_score_above_threshold(self, trained_model):
        """Le F1-score doit être > 0.75."""
        from sklearn.metrics import f1_score
        model, X_test, y_test = trained_model
        y_pred = model.predict(X_test)
        f1     = f1_score(y_test, y_pred, average="weighted")
        assert f1 > 0.75, f"F1-score trop faible : {f1:.4f} (min: 0.75)"


# ─────────────────────────────────────────
#  TESTS — Sécurité & Robustesse
# ─────────────────────────────────────────
class TestRobustness:

    def test_model_handles_zeros(self, trained_model):
        """Le modèle ne plante pas avec des features à zéro."""
        model, _, _ = trained_model
        X_zeros = np.zeros((1, 20))
        pred = model.predict(X_zeros)
        assert pred[0] in [0, 1]

    def test_model_handles_large_values(self, trained_model):
        """Le modèle ne plante pas avec de très grandes valeurs."""
        model, _, _ = trained_model
        X_large = np.ones((1, 20)) * 1e6
        pred = model.predict(X_large)
        assert pred[0] in [0, 1]

    def test_model_handles_negative_values(self, trained_model):
        """Le modèle ne plante pas avec des valeurs négatives (après scaling)."""
        model, _, _ = trained_model
        X_neg = np.ones((1, 20)) * -100
        pred = model.predict(X_neg)
        assert pred[0] in [0, 1]

    def test_false_positive_rate_acceptable(self, trained_model):
        """Le taux de faux positifs doit être < 30% sur les données de test."""
        model, X_test, y_test = trained_model
        y_pred = model.predict(X_test)

        benign_mask = y_test == 0
        if benign_mask.sum() > 0:
            fpr = (y_pred[benign_mask] == 1).mean()
            assert fpr < 0.30, f"Trop de faux positifs : {fpr:.2%} (max: 30%)"

    def test_attack_detection_rate(self, trained_model):
        """Le recall sur les attaques doit être > 50%."""
        model, X_test, y_test = trained_model
        y_pred = model.predict(X_test)

        attack_mask = y_test == 1
        if attack_mask.sum() > 0:
            recall = (y_pred[attack_mask] == 1).mean()
            assert recall > 0.50, f"Recall attaques trop faible : {recall:.2%} (min: 50%)"

    def test_model_not_all_benign(self, trained_model):
        """Le modèle ne doit pas prédire uniquement BENIGN (biais de classe)."""
        model, X_test, _ = trained_model
        preds = model.predict(X_test)
        attack_rate = preds.mean()
        assert attack_rate > 0.01, \
            f"Modèle trop biaisé vers BENIGN : seulement {attack_rate:.2%} d'attaques prédites"

    def test_model_not_all_attack(self, trained_model):
        """Le modèle ne doit pas prédire uniquement ATTACK."""
        model, X_test, _ = trained_model
        preds = model.predict(X_test)
        benign_rate = (preds == 0).mean()
        assert benign_rate > 0.01, \
            f"Modèle trop biaisé vers ATTACK : seulement {benign_rate:.2%} de BENIGN prédit"


# ─────────────────────────────────────────
#  TESTS — Reproductibilité
# ─────────────────────────────────────────
class TestReproducibility:

    def test_deterministic_predictions(self, trained_model):
        """Les prédictions doivent être déterministes (même input → même output)."""
        model, X_test, _ = trained_model
        sample = X_test[:10]

        pred1 = model.predict(sample)
        pred2 = model.predict(sample)

        np.testing.assert_array_equal(pred1, pred2, "Prédictions non déterministes")

    def test_model_has_feature_importances(self, trained_model):
        """Le modèle doit avoir des importances de features non nulles."""
        model, _, _ = trained_model
        importances = model.feature_importances_
        assert len(importances) > 0, "Pas d'importances de features"
        assert importances.sum() > 0, "Toutes les importances sont nulles"


# ─────────────────────────────────────────
#  TESTS — Intégration avec le modèle sauvegardé
# ─────────────────────────────────────────
class TestSavedModel:

    def test_saved_model_exists(self):
        """Au moins un fichier de modèle doit exister après l'entraînement."""
        models_dir = Path(__file__).resolve().parents[2] / "models" / "saved"
        if not models_dir.exists():
            pytest.skip("Dossier models/saved non trouvé — lancer d'abord train.py")

        model_files = list(models_dir.glob("*.joblib"))
        assert len(model_files) > 0, \
            f"Aucun modèle trouvé dans {models_dir}. Lancer : python Model/src/models/train.py"

    def test_saved_model_loadable(self):
        """Le modèle sauvegardé doit être chargeable."""
        import joblib
        models_dir = Path(__file__).resolve().parents[2] / "models" / "saved"
        model_files = list(models_dir.glob("*.joblib"))

        if not model_files:
            pytest.skip("Aucun modèle sauvegardé trouvé")

        model = joblib.load(model_files[0])
        assert model is not None
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")
