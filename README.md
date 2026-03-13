# SecuredMLOps
Conception et Implémentation d'un Pipeline MLOps Sécurisé Basé sur l'Approche DevSecOps

# 🚀 Guide d'Exécution — Pipeline ML IDS CICIDS2017

## Structure finale du projet

```
SECUREDMLOPS/
├── Model/
│   ├── data/
│   │   ├── raw/                    ← Placer les CSV CICIDS2017 ici
│   │   └── processed/              ← Généré automatiquement
│   ├── models/
│   │   └── saved/                  ← Modèles entraînés
│   ├── docs/                       ← Graphiques et rapports
│   ├── notebooks/
│   │   └── 01_EDA.ipynb            ← Exploration des données
│   ├── src/
│   │   ├── data/
│   │   │   ├── download.py         ← Étape 1 : Vérification dataset
│   │   │   ├── validate.py         ← Étape 2 : Validation qualité
│   │   │   ├── preprocess.py       ← Étape 3 : Nettoyage
│   │   │   └── balance.py          ← Étape 4 : SMOTE
│   │   ├── models/
│   │   │   ├── train.py            ← Étape 5 : Entraînement
│   │   │   ├── evaluate.py         ← Étape 6 : Évaluation
│   │   │   ├── optimize.py         ← Étape 7 : Optuna (optionnel)
│   │   │   └── explain.py          ← Étape 8 : SHAP
│   │   └── api/
│   │       └── main.py             ← Étape 9 : API FastAPI
│   └── tests/
│       ├── test_preprocess.py      ← Tests preprocessing
│       └── test_model.py           ← Tests modèle
├── params.yaml                     ← Configuration centrale
├── requirements.txt                ← Dépendances Python
├── Dockerfile                      ← Image Docker
└── docker-compose.yml              ← Stack complète
```

---

## ⚙️ INSTALLATION (une seule fois)

### Étape 0.1 — Créer un environnement virtuel Python

Ouvrir un terminal dans VS Code (`Ctrl + ù`) et exécuter :

```bash
# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement (Windows)
venv\Scripts\activate

# Vérifier que c'est activé (tu dois voir "(venv)" dans le terminal)
python --version
```

### Étape 0.2 — Installer les dépendances

```bash
pip install -r requirements.txt
```

> ⏳ Cette étape prend 3-5 minutes la première fois.

---

## 📁 ÉTAPE 1 — Télécharger le Dataset

```bash
python Model/src/data/download.py
```

Ce script va t'afficher les instructions pour télécharger CICIDS2017.

**Après téléchargement**, place les fichiers CSV dans `Model/data/raw/`
puis relance :
```bash
python Model/src/data/download.py
```
✅ Tu dois voir : `Dataset prêt ! Prochaine étape : python Model/src/data/preprocess.py`

---

## 📊 ÉTAPE 2 — Explorer les Données (EDA)

Lance Jupyter Notebook :
```bash
jupyter notebook Model/notebooks/01_EDA.ipynb
```

Exécute toutes les cellules dans l'ordre. Ce notebook génère des graphiques
dans `Model/docs/` qui seront utiles pour ton rapport.

---

## ✅ ÉTAPE 3 — Valider la Qualité des Données

```bash
python Model/src/data/validate.py
```

**Output attendu :**
```
R1_colonnes_essentielles         ✅ PASS
R2_label_non_null                ✅ PASS
R3_labels_valides                ✅ PASS
R4_pas_de_inf                    ✅ PASS   (ou ❌ si inf présents — normal)
R5_taux_nan                      ✅ PASS
R6_flow_duration_positif         ✅ PASS
R7_nombre_colonnes               ✅ PASS
R8_taille_dataset                ✅ PASS
R9_desequilibre_classes          ✅ PASS   (info seulement)
```

Un rapport HTML est généré : `Model/data/validation_report.html`
Ouvre-le dans ton navigateur pour voir le rapport visuel.

---

## 🧹 ÉTAPE 4 — Prétraiter les Données

```bash
python Model/src/data/preprocess.py
```

**Ce que ça fait :**
- Fusionne tous les CSV (lundi → vendredi)
- Supprime les doublons, valeurs infinies, NaN
- Supprime les colonnes identifiants (IP, Port...)
- Encode les labels (BENIGN=0, ATTACK=1)
- Split stratifié Train/Val/Test (70/10/20)
- Normalise avec StandardScaler

**Output attendu dans `Model/data/processed/` :**
```
X_train.npy          ← Features d'entraînement
X_val.npy            ← Features de validation
X_test.npy           ← Features de test
y_train.npy          ← Labels d'entraînement
y_val.npy            ← Labels de validation
y_test.npy           ← Labels de test
feature_names.txt    ← Noms des features
scaler.joblib        ← Scaler sauvegardé (pour l'API)
```

> ⏳ Cette étape peut prendre 5-15 minutes selon ta RAM (dataset ~2.8GB).

---

## ⚖️ ÉTAPE 5 — Rééquilibrer les Classes (SMOTE)

```bash
python Model/src/data/balance.py
```

**Ce que ça fait :**
- Applique SMOTE pour créer des exemples synthétiques d'attaques
- Rééquilibre le dataset d'entraînement (était 80% BENIGN, devient ~50/50)

**Output :**
```
X_train_balanced.npy
y_train_balanced.npy
```

---

## 🏋️ ÉTAPE 6 — Entraîner le Modèle

**Ouvrir un terminal séparé et lancer MLflow :**
```bash
mlflow ui --port 5000
```
Laisser ce terminal ouvert. Aller sur http://localhost:5000 dans le navigateur.

**Dans un autre terminal, lancer l'entraînement :**
```bash
python Model/src/models/train.py
```

**Output attendu :**
- Modèle Baseline (Logistic Regression) entraîné et loggé
- Modèle XGBoost entraîné avec early stopping
- Métriques affichées dans le terminal
- Graphiques générés dans `Model/docs/`
- Modèle visible dans http://localhost:5000

**Fichiers générés :**
```
Model/models/saved/xgboost_ids.joblib
Model/docs/confusion_matrix_val.png
Model/docs/confusion_matrix_test.png
Model/docs/classification_report.txt
Model/docs/feature_importance.png
```

---

## 📈 ÉTAPE 7 — Évaluer le Modèle

```bash
python Model/src/models/evaluate.py
```

**Output :** Rapport HTML complet → `Model/docs/evaluation_report.html`

**Métriques cibles :**
```
✅ F1 > 0.97
✅ ROC AUC > 0.99
✅ FPR < 5%
✅ Recall > 95%
```

---

## 🔧 ÉTAPE 8 — Optimiser les Hyperparamètres (Optionnel)

```bash
python Model/src/models/optimize.py
```

Lance 50 essais Optuna pour trouver les meilleurs hyperparamètres.
> ⏳ Peut prendre 30-60 minutes selon ta machine.

---

## 💡 ÉTAPE 9 — Explicabilité SHAP

```bash
python Model/src/models/explain.py
```

**Graphiques générés :**
```
Model/docs/shap_summary_bar.png     ← Importance globale
Model/docs/shap_beeswarm.png        ← Distribution des impacts
Model/docs/shap_force_attack.png    ← Explication d'une attaque
Model/docs/shap_force_benign.png    ← Explication d'un trafic normal
Model/docs/shap_dependence.png      ← Dépendance feature principale
```

---

## 🧪 ÉTAPE 10 — Lancer les Tests

```bash
# Tests de preprocessing uniquement
pytest Model/tests/test_preprocess.py -v

# Tests du modèle uniquement
pytest Model/tests/test_model.py -v

# Tous les tests avec couverture
pytest Model/tests/ -v --cov=Model/src --cov-report=html
```

**Output attendu :** Tous les tests en ✅ PASSED

---

## 🌐 ÉTAPE 11 — Lancer l'API

```bash
uvicorn Model.src.api.main:app --reload --port 8000
```

**URLs disponibles :**
- API : http://localhost:8000
- Documentation interactive : http://localhost:8000/docs
- Health check : http://localhost:8000/health
- Infos modèle : http://localhost:8000/model/info

**Tester l'API avec curl :**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [120.0, 2, 1, 5000.0, 25.0, 40.0, 0.0, 15.5, 10.0, 5.0]}'
```

---

## 🐳 ÉTAPE 12 — Docker (optionnel)

```bash
# Builder l'image
docker build -t ids-api:1.0 .

# Lancer le conteneur
docker run -p 8000:8000 ids-api:1.0

# Ou avec docker-compose (MLflow + API ensemble)
docker-compose up -d
```

---

## ❓ Problèmes Fréquents

| Problème | Solution |
|---|---|
| `ModuleNotFoundError` | Vérifier que le venv est activé : `venv\Scripts\activate` |
| `FileNotFoundError: aucun CSV` | Placer les CSV dans `Model/data/raw/` |
| `MLflow connection refused` | Lancer d'abord `mlflow ui --port 5000` |
| `MemoryError` | Réduire la taille du dataset dans `preprocess.py` (ajouter `nrows=500000`) |
| Tests qui échouent `test_saved_model` | Lancer d'abord `train.py` |
