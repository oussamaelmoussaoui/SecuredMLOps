# 🔐 SecuredMLOps

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=for-the-badge)
![MLflow](https://img.shields.io/badge/MLflow-2.8-blue?style=for-the-badge&logo=mlflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-24.x-blue?style=for-the-badge&logo=docker)
![DVC](https://img.shields.io/badge/DVC-3.x-purple?style=for-the-badge)

**Conception et Implémentation d'un Pipeline MLOps Sécurisé Basé sur l'Approche DevSecOps**

*Projet de Fin d'Année — Data Science & Cloud Computing*

</div>

---

## 📋 Table des Matières

- [À propos du projet](#-à-propos-du-projet)
- [Objectifs](#-objectifs)
- [Architecture globale](#-architecture-globale)
- [Stack technologique](#-stack-technologique)
- [Résultats du modèle ML](#-résultats-du-modèle-ml)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Récupérer les données DVC](#-récupérer-les-données-dvc)
- [Exécution du pipeline](#-exécution-du-pipeline)
- [Tests](#-tests)
- [API FastAPI](#-api-fastapi)
- [Docker](#-docker)
- [Structure du projet](#-structure-du-projet)
- [Problèmes fréquents](#-problèmes-fréquents)
- [Équipe](#-équipe)

---

## 🎯 À propos du projet

**SecuredMLOps** est un pipeline MLOps complet et sécurisé qui intègre les pratiques **DevSecOps** à chaque étape du cycle de vie d'un modèle de Machine Learning.

### Le problème résolu

La majorité des projets ML en entreprise souffrent de trois problèmes majeurs :

- **Absence de standardisation** — chaque data scientist déploie ses modèles différemment, rendant la collaboration et la maintenance impossibles
- **Sécurité négligée** — les vulnérabilités et les données sensibles exposées sont découvertes trop tard, en production
- **Manque de traçabilité** — impossible de savoir quelle version du modèle tourne en production, sur quelles données il a été entraîné, et pourquoi il se dégrade

### Notre solution

Ce projet construit une **infrastructure automatisée, reproductible et sécurisée** qui gère l'intégralité du cycle de vie ML en appliquant le principe **Shift-Left Security** : la sécurité est intégrée dès le développement, jamais ajoutée à la fin.

Le cas d'usage concret choisi est la **détection d'intrusions réseau** — un domaine où la sécurité du pipeline MLOps et la sécurité applicative se rejoignent naturellement, en utilisant le dataset de référence académique **CICIDS2017** (Canadian Institute for Cybersecurity).

### Valeur professionnelle

Ce type de projet répond directement aux exigences de l'industrie et de la réglementation actuelle :

- **EU AI Act (2024)** — impose la traçabilité et l'explicabilité des systèmes d'IA critiques
- **Profils MLOps Engineer** — parmi les plus recherchés avec des salaires démarrant à 70k€ en Europe
- **Secteur cybersécurité + ML** — niche très rare et donc très bien rémunérée

---

## 🏆 Objectifs

| Objectif | Description |
|----------|-------------|
| **Automatisation complète** | Pipeline end-to-end : Data → Model → Deploy → Monitor |
| **Security-as-Code** | Sécurité intégrée à chaque étape (DevSecOps) |
| **Traçabilité totale** | Versioning données (DVC), modèles (MLflow), code (Git) |
| **Reproductibilité** | Tout membre de l'équipe reproduit les mêmes résultats |
| **Détection de drift** | Alertes automatiques si le modèle se dégrade en production |
| **Explicabilité** | Chaque prédiction est justifiable via SHAP |

---

## 🏗 Architecture Globale

Le pipeline est organisé en **6 phases séquentielles** sur 20 semaines :

```
┌─────────────────────────────────────────────────────────────┐
│                     SECUREDMLOPS PIPELINE                   │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│ Phase 1  │ Phase 2  │ Phase 3  │ Phase 4  │ Phase 5 & 6    │
│  Setup   │  Data    │   ML     │  CI/CD   │ Deploy+Monitor │
│ S1 → S3  │ S4 → S6  │ S7 → S9  │ S10→ S13 │  S14 → S20    │
└──────────┴──────────┴──────────┴──────────┴────────────────┘
```

**Flux de données :**
```
CICIDS2017 (8 CSV, 843 MB)
          ↓
 Validation & Nettoyage
          ↓
 SMOTE — Rééquilibrage
          ↓
 XGBoost Training  ←→  MLflow Tracking
          ↓
 SHAP Explainability
          ↓
 FastAPI  →  Docker  →  Kubernetes (Phase 5)
          ↓
 Prometheus + Grafana + Evidently (Phase 6)
```

**Statut des phases :**

| Phase | Contenu | Statut |
|-------|---------|--------|
| Phase 1 — Setup | Environnement, DVC, MLflow, architecture | ✅ Terminé |
| Phase 2 — Data | Ingestion, validation, versioning DVC | ✅ Terminé |
| Phase 3 — ML | XGBoost, MLflow, SHAP, API FastAPI, Tests | ✅ Terminé |
| Phase 4 — CI/CD + Sécurité | GitHub Actions, Bandit, Trivy, Vault | 🔄 En cours |
| Phase 5 — Déploiement | Kubernetes, Canary, ArgoCD | ⬜ À venir |
| Phase 6 — Monitoring | Grafana, Evidently drift, rapport PFA | ⬜ À venir |

---

## 🛠 Stack Technologique

### ML & Data Science
| Outil | Rôle |
|-------|------|
| **XGBoost** | Modèle principal de détection d'intrusions |
| **scikit-learn** | Preprocessing, métriques, split stratifié |
| **imbalanced-learn** | SMOTE — rééquilibrage des classes |
| **SHAP** | Explicabilité des prédictions |
| **Optuna** | Optimisation automatique des hyperparamètres |
| **pandas / numpy** | Manipulation et transformation des données |

### MLOps & Versioning
| Outil | Rôle |
|-------|------|
| **MLflow** | Tracking des expériences + Model Registry |
| **DVC** | Versioning des données volumineuses (843 MB) |
| **DagsHub** | Remote storage DVC + interface MLOps collaborative |
| **GitHub** | Versioning du code source |

### API & Déploiement
| Outil | Rôle |
|-------|------|
| **FastAPI** | API REST pour le serving du modèle |
| **Docker** | Conteneurisation de l'application |
| **docker-compose** | Stack locale MLflow + API |
| **Kubernetes** | Orchestration cloud (Phase 5) |

### CI/CD & Sécurité *(Phase 4)*
| Outil | Rôle |
|-------|------|
| **GitHub Actions** | Pipelines CI/CD automatisés |
| **Bandit** | Analyse statique du code Python (SAST) |
| **Trivy** | Scan des images Docker (CVE) |
| **GitLeaks** | Détection de secrets exposés dans le code |
| **OWASP ZAP** | Tests dynamiques sur l'API (DAST) |
| **HashiCorp Vault** | Gestion centralisée des secrets |

### Monitoring *(Phase 6)*
| Outil | Rôle |
|-------|------|
| **Prometheus** | Collecte métriques système et API |
| **Grafana** | Dashboards de visualisation temps réel |
| **Evidently AI** | Détection data drift et concept drift |

---

## 📊 Résultats du Modèle ML

Modèle XGBoost entraîné sur **2,682,036 flux réseau** du dataset CICIDS2017 :

| Métrique | Valeur | Cible | Statut |
|----------|--------|-------|--------|
| Accuracy | **99.86%** | > 97% | ✅ |
| F1-Score | **0.9986** | > 0.97 | ✅ |
| ROC-AUC | **1.0000** | > 0.99 | ✅ |
| False Positive Rate | **0.16%** | < 5% | ✅ |
| False Negative Rate | **0.04%** | < 5% | ✅ |
| Latence inférence | **< 3ms** | < 100ms | ✅ |
| Tests unitaires | **33 / 33** | 100% | ✅ |

---

## ⚙️ Prérequis

- **Python 3.10+** — [télécharger](https://www.python.org/downloads/)
- **Git** — [télécharger](https://git-scm.com/downloads)
- **Docker Desktop** — [télécharger](https://www.docker.com/products/docker-desktop/)
- **8 GB RAM minimum** (16 GB recommandé)
- **5 GB d'espace disque libre**

---

## 🚀 Installation

### 1. Cloner le repo

```bash
git clone https://github.com/oelmoussawi/SecuredMLOps.git
cd SecuredMLOps
```

### 2. Créer et activer l'environnement virtuel

```powershell
# Créer
python -m venv venv

# Activer (Windows PowerShell)
.\venv\Scripts\Activate.ps1
```

> **Si erreur de politique d'exécution :**
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

Tu dois voir `(venv)` au début de la ligne de commande.

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

> ⏳ 3 à 5 minutes la première fois.

---

## 📦 Récupérer les Données DVC

> ⚠️ **Pour les coéquipiers** — Tu n'as **PAS** besoin de télécharger le dataset manuellement. Les données sont versionnées sur DagsHub.

### Récupérer les données

```bash
dvc pull
```

Cette commande télécharge automatiquement :
- `Model/data/raw/` — les 8 fichiers CSV CICIDS2017 (~843 MB)
- `Model/data/processed/` — les données déjà preprocessées (fichiers `.npy`)

### Vérifier

```powershell
dir Model\data\raw\        # Windows
ls Model/data/raw/         # Linux / macOS
```

✅ Tu dois voir **8 fichiers CSV** dans `raw/` et **8 fichiers** dans `processed/`.

### Si erreur d'authentification

```bash
dvc remote modify dagshub --local auth basic
dvc remote modify dagshub --local user TON_USERNAME_DAGSHUB
dvc remote modify dagshub --local password TON_TOKEN_DAGSHUB
dvc pull
```

> Obtenir ton token : [dagshub.com](https://dagshub.com) → Settings → Access Tokens

### Si erreur "fichiers manquants"

```bash
git pull origin main
dvc pull --force
```

---

## ▶️ Exécution du Pipeline

> **Note :** Les données étant déjà preprocessées via `dvc pull`, tu peux aller **directement à l'Étape 3**. Les étapes 1 et 2 sont optionnelles.

---

### [Optionnel] Étape 1 — Valider la qualité des données

```bash
python Model/src/data/validate.py
```

Génère : `Model/data/validation_report.html`

---

### [Optionnel] Étape 2 — Reproductibilité du preprocessing

> Uniquement si tu veux reproduire le preprocessing depuis les CSV bruts.

```bash
# Nettoyage et normalisation
python Model/src/data/preprocess.py

# Rééquilibrage SMOTE
python Model/src/data/balance.py
```

> ⏳ 30 à 45 minutes au total.

---

### Étape 3 — Lancer MLflow *(terminal séparé)*

Ouvre un **nouveau terminal**, active le venv, puis :

```bash
mlflow ui --port 5000
```

Laisse ce terminal **ouvert**. Interface : **http://localhost:5000**

---

### Étape 4 — Entraîner le modèle

```bash
python Model/src/models/train.py
```

**Fichiers générés :**
```
Model/models/saved/xgboost_ids.joblib
Model/docs/confusion_matrix_val.png
Model/docs/confusion_matrix_test.png
Model/docs/classification_report.txt
Model/docs/feature_importance.png
```

> ⏳ 15 à 30 minutes.

---

### Étape 5 — Évaluer le modèle

```bash
python Model/src/models/evaluate.py
```

Génère : `Model/docs/evaluation_report.html`

---

### Étape 6 — Explicabilité SHAP

```bash
python Model/src/models/explain.py
```

Génère 5 graphiques dans `Model/docs/` expliquant les décisions du modèle.

> ⏳ 5 à 10 minutes.

---

### [Optionnel] Étape 7 — Optimiser les hyperparamètres

```bash
python Model/src/models/optimize.py
```

> ⏳ 30 à 60 minutes.

---

## 🧪 Tests

```bash
# Tous les tests
pytest Model/tests/ -v

# Avec couverture de code
pytest Model/tests/ -v --cov=Model/src --cov-report=html
```

**Résultat attendu :** `33 passed, 2 skipped`

---

## 🌐 API FastAPI

### Lancer l'API

```bash
uvicorn Model.src.api.main:app --reload --port 8000
```

**URLs :**
- Documentation interactive : **http://localhost:8000/docs**
- Health check : **http://localhost:8000/health**
- Infos modèle : **http://localhost:8000/model/info**

### Tester

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [120.0, 2, 1, 5000.0, 25.0, 40.0, 0.0, 15.5, 10.0, 5.0]}'
```

**Réponse :**
```json
{
  "prediction": 1,
  "label": "ATTACK",
  "confidence": 0.9986,
  "inference_time_ms": 2.3
}
```

---

## 🐳 Docker

```bash
# Builder l'image
docker build -t ids-api:1.0 .

# Lancer l'API seule
docker run -p 8000:8000 ids-api:1.0

# Stack complète (MLflow + API)
docker-compose up -d

# Arrêter
docker-compose down
```

---

## 📁 Structure du Projet

```
SecuredMLOps/
├── Model/
│   ├── data/
│   │   ├── raw/                ← CSV CICIDS2017 (DVC → DagsHub)
│   │   └── processed/          ← Données preprocessées (DVC)
│   ├── models/saved/           ← Modèles entraînés (.joblib)
│   ├── docs/                   ← Graphiques et rapports
│   ├── notebooks/01_EDA.ipynb  ← Analyse exploratoire
│   ├── src/
│   │   ├── data/               ← download, validate, preprocess, balance
│   │   ├── models/             ← train, evaluate, optimize, explain
│   │   └── api/main.py         ← API REST FastAPI
│   └── tests/                  ← 33 tests unitaires
├── .dvc/                       ← Configuration DVC
├── .github/workflows/          ← CI/CD GitHub Actions
├── params.yaml                 ← Configuration centralisée
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## ❓ Problèmes Fréquents

| Problème | Solution |
|----------|----------|
| `Activate.ps1` bloqué | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `ModuleNotFoundError` | Vérifier que `(venv)` est visible dans le terminal |
| `dvc pull` — fichiers manquants | `git pull origin main` puis `dvc pull --force` |
| `dvc pull` — timeout DNS | Problème réseau (université). Réessayer depuis une autre connexion |
| `MLflow connection refused` | Lancer `mlflow ui --port 5000` dans un terminal séparé d'abord |
| `MemoryError` pendant train.py | Fermer les autres applications — 8 GB RAM minimum requis |
| Tests `test_saved_model` skipped | Normal — lancer `train.py` d'abord |
| API `503 Service Unavailable` | Vérifier que `train.py` a bien tourné et généré le `.joblib` |

---

## 👥 Équipe

| Membre | Rôle | GitHub |
|--------|------|--------|
| EL MOUSSAOUI Oussama | ML Pipeline & MLOps Lead | [@oelmoussawi](https://github.com/oelmoussawi) |
| — | — | — |
| — | — | — |

**Liens :**
- 🐙 GitHub : [oelmoussawi/SecuredMLOps](https://github.com/oelmoussawi/SecuredMLOps)
- 📊 DagsHub : [oelmoussawi/SecuredMLOps](https://dagshub.com/oelmoussawi/SecuredMLOps)
- 📖 Dataset : [CICIDS2017 — University of New Brunswick](https://www.unb.ca/cic/datasets/ids-2017.html)

---

<div align="center">
<i>Projet de Fin d'Année — Filière Data Science & Cloud Computing</i>
</div>
