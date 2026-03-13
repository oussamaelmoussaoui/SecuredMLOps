# ─────────────────────────────────────────
#  Model/src/api/main.py
#  API FastAPI — Serving du modèle IDS
# ─────────────────────────────────────────
#
#  COMMENT UTILISER :
#  > uvicorn Model.src.api.main:app --reload --port 8000
#
#  Documentation interactive :
#  http://localhost:8000/docs
# ─────────────────────────────────────────

import logging
import time
from pathlib import Path
from typing import List

import joblib
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[3]
MODELS_DIR    = PROJECT_ROOT / "Model" / "models" / "saved"
PROCESSED_DIR = PROJECT_ROOT / "Model" / "data" / "processed"
PARAMS_FILE   = PROJECT_ROOT / "params.yaml"

# ─────────────────────────────────────────
#  APP
# ─────────────────────────────────────────
app = FastAPI(
    title="🔐 Network Intrusion Detection API",
    description="""
## API de Détection d'Intrusions Réseau
Modèle XGBoost entraîné sur le dataset CICIDS2017.

### Endpoints
- `POST /predict`       — Prédire si un flux réseau est une intrusion
- `POST /predict/batch` — Prédiction sur un lot de flux
- `GET  /health`        — Statut de l'API
- `GET  /model/info`    — Informations sur le modèle
    """,
    version="1.0.0",
)

# ─────────────────────────────────────────
#  ÉTAT GLOBAL
# ─────────────────────────────────────────
model       = None
scaler      = None
feat_names  = None
params      = None


@app.on_event("startup")
async def load_model():
    """Charger le modèle et le scaler au démarrage."""
    global model, scaler, feat_names, params

    with open(PARAMS_FILE) as f:
        params = yaml.safe_load(f)

    # Charger feature names
    feat_path = PROCESSED_DIR / "feature_names.txt"
    if feat_path.exists():
        with open(feat_path) as f:
            feat_names = f.read().splitlines()

    # Charger le scaler
    scaler_path = PROCESSED_DIR / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        logger.info("Scaler chargé")

    # Charger le modèle (optimisé > base)
    optimized = MODELS_DIR / "xgboost_ids_optimized.joblib"
    base      = MODELS_DIR / "xgboost_ids.joblib"

    if optimized.exists():
        model = joblib.load(optimized)
        logger.info(f"Modèle optimisé chargé : {optimized.name}")
    elif base.exists():
        model = joblib.load(base)
        logger.info(f"Modèle de base chargé : {base.name}")
    else:
        logger.warning("⚠️  Aucun modèle trouvé. Exécutez d'abord : python Model/src/models/train.py")


# ─────────────────────────────────────────
#  SCHÉMAS (Pydantic)
# ─────────────────────────────────────────
class NetworkFlow(BaseModel):
    """
    Représentation d'un flux réseau CICIDS2017.
    Envoyer les valeurs des features dans le même ordre
    que feature_names.txt
    """
    features: List[float] = Field(
        ...,
        description="Liste des valeurs des features réseau (même ordre que feature_names.txt)",
        example=[0.0, 1.0, 0.5, 120.0, 0.0, 0.0, 40.0, 40.0, 0.0, 0.0]
    )

    class Config:
        schema_extra = {
            "example": {
                "features": [120.0, 2, 1, 5000.0, 25.0, 40.0, 0.0, 15.5, 10.0, 5.0]
            }
        }


class PredictionResponse(BaseModel):
    prediction:       int    = Field(..., description="0=BENIGN, 1=ATTACK")
    label:            str    = Field(..., description="BENIGN ou ATTACK")
    confidence:       float  = Field(..., description="Probabilité de la prédiction (0-1)")
    attack_probability: float = Field(..., description="Probabilité d'être une attaque")
    inference_time_ms: float  = Field(..., description="Temps d'inférence en millisecondes")


class BatchRequest(BaseModel):
    flows: List[NetworkFlow] = Field(..., description="Liste de flux réseau à analyser")


class BatchResponse(BaseModel):
    predictions:  List[PredictionResponse]
    total:        int
    attacks_found: int
    processing_time_ms: float


# ─────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────

@app.get("/health", tags=["Monitoring"])
async def health():
    """Vérifier l'état de l'API."""
    return {
        "status":       "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "n_features":   len(feat_names) if feat_names else None,
    }


@app.get("/model/info", tags=["Monitoring"])
async def model_info():
    """Informations sur le modèle en production."""
    return {
        "model_type":       "XGBoost Classifier",
        "training_dataset": "CICIDS2017",
        "task":             "Binary Classification (BENIGN vs ATTACK)",
        "features":         feat_names,
        "n_features":       len(feat_names) if feat_names else None,
        "classes":          {0: "BENIGN", 1: "ATTACK"},
        "version":          "1.0.0",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(flow: NetworkFlow):
    """
    Prédire si un flux réseau est une intrusion.

    Envoyer les features dans le même ordre que `/model/info` → `features`.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non chargé. Exécutez d'abord l'entraînement."
        )

    if feat_names and len(flow.features) != len(feat_names):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Nombre de features incorrect : reçu {len(flow.features)}, attendu {len(feat_names)}"
        )

    try:
        start = time.time()

        X = np.array(flow.features).reshape(1, -1)

        # Appliquer le scaler si disponible
        if scaler is not None:
            X = scaler.transform(X)

        prediction      = int(model.predict(X)[0])
        probabilities   = model.predict_proba(X)[0]
        confidence      = float(probabilities[prediction])
        attack_prob     = float(probabilities[1])
        inference_ms    = (time.time() - start) * 1000

        logger.info(
            f"Prédiction : {'ATTACK' if prediction == 1 else 'BENIGN'} | "
            f"Confiance : {confidence:.2%} | "
            f"Temps : {inference_ms:.2f}ms"
        )

        return PredictionResponse(
            prediction=prediction,
            label="ATTACK" if prediction == 1 else "BENIGN",
            confidence=round(confidence, 4),
            attack_probability=round(attack_prob, 4),
            inference_time_ms=round(inference_ms, 2),
        )

    except Exception as e:
        logger.error(f"Erreur de prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(request: BatchRequest):
    """Prédiction sur un lot de flux réseau."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    start = time.time()
    predictions = []

    X_batch = np.array([f.features for f in request.flows])
    if scaler is not None:
        X_batch = scaler.transform(X_batch)

    batch_preds  = model.predict(X_batch)
    batch_probas = model.predict_proba(X_batch)

    for i, (pred, proba) in enumerate(zip(batch_preds, batch_probas)):
        predictions.append(PredictionResponse(
            prediction=int(pred),
            label="ATTACK" if pred == 1 else "BENIGN",
            confidence=round(float(proba[pred]), 4),
            attack_probability=round(float(proba[1]), 4),
            inference_time_ms=0.0,
        ))

    total_ms    = (time.time() - start) * 1000
    attacks     = sum(1 for p in predictions if p.prediction == 1)

    logger.info(f"Batch : {len(request.flows)} flux | {attacks} attaques détectées | {total_ms:.2f}ms")

    return BatchResponse(
        predictions=predictions,
        total=len(predictions),
        attacks_found=attacks,
        processing_time_ms=round(total_ms, 2),
    )
