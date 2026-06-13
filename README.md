# 🔐 SecuredMLOps

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=for-the-badge)
![MLflow](https://img.shields.io/badge/MLflow-2.8-blue?style=for-the-badge&logo=mlflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-24.x-blue?style=for-the-badge&logo=docker)
![Kubernetes](https://img.shields.io/badge/Kubernetes-1.29+-326CE5?style=for-the-badge&logo=kubernetes)
![Grafana](https://img.shields.io/badge/Grafana-10.x-F46800?style=for-the-badge&logo=grafana)
![Evidently](https://img.shields.io/badge/Evidently-0.4+-purple?style=for-the-badge)

**Conception et Implémentation d'un Pipeline MLOps Sécurisé Basé sur l'Approche DevSecOps**

*Projet de Fin d'Année — Data Science & Cloud Computing*

</div>

---

## 📋 Table des Matières

- [À propos du projet](#-à-propos-du-projet)
- [Architecture globale](#-architecture-globale)
- [Stack technologique](#-stack-technologique)
- [Résultats du modèle ML](#-résultats-du-modèle-ml)
- [Prérequis](#-prérequis)
- [Installation & Démarrage rapide](#-installation--démarrage-rapide)
- [Pipeline ML — Phases 1 à 3](#-pipeline-ml--phases-1-à-3)
- [CI/CD & Sécurité — Phase 4](#-cicd--sécurité--phase-4)
- [Déploiement Kubernetes — Phase 5](#-déploiement-kubernetes--phase-5)
- [Monitoring — Phase 6](#-monitoring--phase-6)
- [Structure du projet](#-structure-du-projet)
- [Problèmes fréquents](#-problèmes-fréquents)
- [Équipe](#-équipe)

---

## 🎯 À propos du projet

**SecuredMLOps** est un pipeline MLOps complet et sécurisé qui intègre les pratiques **DevSecOps** à chaque étape du cycle de vie d'un modèle de Machine Learning, de la donnée brute jusqu'au monitoring en production.

### Le problème résolu

La majorité des projets ML en entreprise souffrent de trois problèmes majeurs :

- **Absence de standardisation** — chaque data scientist déploie ses modèles différemment, rendant la collaboration et la maintenance impossibles
- **Sécurité négligée** — les vulnérabilités et les données sensibles exposées sont découvertes trop tard, en production
- **Manque de traçabilité** — impossible de savoir quelle version du modèle tourne en production, sur quelles données il a été entraîné, et pourquoi il se dégrade

### Notre solution

Ce projet construit une **infrastructure automatisée, reproductible et sécurisée** qui gère l'intégralité du cycle de vie ML en appliquant le principe **Shift-Left Security** : la sécurité est intégrée dès le développement, jamais ajoutée à la fin.

Le cas d'usage concret est la **détection d'intrusions réseau**, en utilisant le dataset académique **CICIDS2017** (Canadian Institute for Cybersecurity).

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

**Flux de données complet :**
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
 FastAPI  →  Docker  →  Kubernetes (Canary via Argo Rollouts)
                                    ↓
                    Prometheus + Grafana + Evidently (Drift)
```

**Statut des phases :**

| Phase | Contenu | Statut |
|-------|---------|--------|
| Phase 1 — Setup | Environnement, DVC, MLflow, architecture | ✅ Terminé |
| Phase 2 — Data | Ingestion, validation, versioning DVC | ✅ Terminé |
| Phase 3 — ML | XGBoost, MLflow, SHAP, API FastAPI, Tests | ✅ Terminé |
| Phase 4 — CI/CD + Sécurité | GitHub Actions, Bandit, Trivy, Vault | 🔄 En cours |
| Phase 5 — Déploiement | Kubernetes, Canary, ArgoCD | ⬜ À venir |
| Phase 6 — Monitoring | Grafana, Evidently drift | ⬜ À venir |

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
| **Kubernetes** | Orchestration cloud (Phase 5) |
| **Helm** | Packaging des manifests Kubernetes |
| **Argo Rollouts** | Stratégie Canary + rollback automatique |
| **ArgoCD** | GitOps — synchronisation Git ↔ cluster |

### CI/CD & Sécurité
| Outil | Rôle |
|-------|------|
| **GitHub Actions** | Pipelines CI/CD automatisés |
| **Bandit** | Analyse statique du code Python (SAST) |
| **Trivy** | Scan des images Docker (CVE) |
| **GitLeaks** | Détection de secrets exposés dans le code |
| **OWASP ZAP** | Tests dynamiques sur l'API (DAST) |
| **HashiCorp Vault** | Gestion centralisée des secrets |

### Monitoring
| Outil | Rôle |
|-------|------|
| **Prometheus** | Collecte métriques système et API |
| **Grafana** | Dashboards de visualisation temps réel |
| **Evidently AI** | Détection data drift et concept drift |

---

## 📊 Résultats du Modèle ML

Modèle XGBoost entraîné sur **2 682 036 flux réseau** du dataset CICIDS2017 :

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

| Outil | Version min | Lien |
|-------|-------------|------|
| Python | 3.10+ | [python.org](https://www.python.org/downloads/) |
| Git | — | [git-scm.com](https://git-scm.com/downloads) |
| Docker Desktop | 24.x | [docker.com](https://www.docker.com/products/docker-desktop/) |
| kubectl | 1.29+ | [kubernetes.io](https://kubernetes.io/docs/tasks/tools/) |
| Minikube | 1.32+ | [minikube.sigs.k8s.io](https://minikube.sigs.k8s.io/) |
| Helm | 3.13+ | [helm.sh](https://helm.sh/docs/intro/install/) |
| kubectl-argo-rollouts | 1.6+ | [argoproj.github.io](https://argoproj.github.io/argo-rollouts/) |
| RAM | 8 GB min (16 GB recommandé) | — |
| Espace disque | 5 GB libre | — |

---

## 🚀 Installation & Démarrage rapide

### 1. Cloner le repo

```bash
git clone https://github.com/oussamaelmoussaoui/SecuredMLOps.git
cd SecuredMLOps
```

### 2. Créer et activer l'environnement virtuel

```powershell
python -m venv venv
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

### 4. Récupérer les données DVC

```bash
dvc pull
```

Cette commande télécharge automatiquement :
- `Model/data/raw/` — les 8 fichiers CSV CICIDS2017 (~843 MB)
- `Model/data/processed/` — les données déjà preprocessées (fichiers `.npy`)

> **Si erreur d'authentification DVC :**
> ```bash
> dvc remote modify dagshub --local auth basic
> dvc remote modify dagshub --local user TON_USERNAME_DAGSHUB
> dvc remote modify dagshub --local password TON_TOKEN_DAGSHUB
> dvc pull
> ```
> Obtenir ton token : [dagshub.com](https://dagshub.com) → Settings → Access Tokens

---

## 🤖 Pipeline ML — Phases 1 à 3

> Les données étant déjà preprocessées via `dvc pull`, tu peux aller directement à l'**Étape 3**. Les étapes 1 et 2 sont optionnelles.

### [Optionnel] Étape 1 — Valider la qualité des données

```bash
python Model/src/data/validate.py
```

Génère : `Model/data/validation_report.html`

### [Optionnel] Étape 2 — Reproductibilité du preprocessing

> Uniquement si tu veux reproduire le preprocessing depuis les CSV bruts (30 à 45 minutes).

```bash
python Model/src/data/preprocess.py
python Model/src/data/balance.py
```

### Étape 3 — Lancer MLflow *(terminal séparé)*

```bash
mlflow ui --port 5000
```

Laisse ce terminal **ouvert**. Interface disponible sur **http://localhost:5000**

### Étape 4 — Entraîner le modèle

```bash
python Model/src/models/train.py
```

Fichiers générés dans `Model/models/saved/` et `Model/docs/` (15 à 30 minutes).

### Étape 5 — Évaluer le modèle

```bash
python Model/src/models/evaluate.py
```

Génère : `Model/docs/evaluation_report.html`

### Étape 6 — Explicabilité SHAP

```bash
python Model/src/models/explain.py
```

Génère 5 graphiques dans `Model/docs/` expliquant les décisions du modèle (5 à 10 minutes).

### Étape 7 — Tests

```bash
pytest Model/tests/ -v
```

Résultat attendu : `33 passed, 2 skipped`

### Étape 8 — Lancer l'API FastAPI

```bash
uvicorn Model.src.api.main:app --reload --port 8000
```

- Documentation interactive : **http://localhost:8000/docs**
- Health check : **http://localhost:8000/health**
- Infos modèle : **http://localhost:8000/model/info**

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [120.0, 2, 1, 5000.0, 25.0, 40.0, 0.0, 15.5, 10.0, 5.0]}'
```

Réponse attendue :
```json
{
  "prediction": 1,
  "label": "ATTACK",
  "confidence": 0.9986,
  "inference_time_ms": 2.3
}
```

### Docker

```bash
docker build -t ids-api:1.0 .
docker run -p 8000:8000 ids-api:1.0

# Stack complète (MLflow + API)
docker-compose up -d
docker-compose down
```

---

## 🔒 CI/CD & Sécurité — Phase 4

Cette phase intègre la sécurité à **chaque étape** via GitHub Actions.

| Outil | Type | Ce qu'il détecte |
|-------|------|-----------------|
| **Bandit** | SAST | Vulnérabilités dans le code Python |
| **Trivy** | Image scan | CVE dans les images Docker |
| **GitLeaks** | Secret scan | Clés API, tokens exposés dans le code |
| **OWASP ZAP** | DAST | Failles sur l'API en cours d'exécution |
| **HashiCorp Vault** | Secrets | Gestion centralisée des credentials |

---

## 🚢 Déploiement Kubernetes — Phase 5

### Architecture Canary

```
                   ┌──────────────────────┐
                   │   Git Repository      │
                   │ (k8s/helm/ids-api/)   │
                   └──────────┬───────────┘
                              │ surveille
                              ▼
                   ┌──────────────────────┐
                   │       ArgoCD         │
                   │   (GitOps engine)    │
                   └──────────┬───────────┘
                              │ applique
                              ▼
                   ┌──────────────────────┐
                   │   Argo Rollouts      │
                   │  (Canary strategy)   │
                   └──────────┬───────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │  Pod v1  │        │  Pod v1  │        │  Pod v2  │
    │ (stable) │        │ (stable) │        │ (canary) │
    └──────────┘        └──────────┘        └──────────┘
                                  ↓
                       ┌──────────────────────┐
                       │   Ingress NGINX      │
                       │  + TLS + Rate Limit  │
                       └──────────────────────┘
```

**Flux Canary progressif :**
```
10% → pause 2min → analyse Prometheus ─┐
25% → pause 2min → analyse Prometheus ─┤  ❌ Échec → rollback auto
50% → pause 5min → analyse Prometheus ─┤
75% → pause 2min → analyse Prometheus ─┘
100% → version stable promue
```

Le rollback est **automatique** si Prometheus détecte :
- success-rate < 95% pendant 2 mesures consécutives
- latence p95 > 100ms pendant 2 mesures consécutives

### Installation locale (Minikube)

```bash
cd k8s/scripts
chmod +x *.sh
./setup-minikube.sh        # Bootstrap complet (~10 min)
./deploy-api.sh            # Build + déploiement Helm
```

### Tester l'API

```bash
kubectl port-forward -n securedmlops svc/ids-api 8000:80
curl http://localhost:8000/health
```

### Déploiement Helm

```bash
# Dev local
helm upgrade --install ids-api k8s/helm/ids-api \
  --namespace securedmlops --create-namespace \
  -f k8s/helm/ids-api/values-dev.yaml

# Production
helm upgrade --install ids-api k8s/helm/ids-api \
  --namespace securedmlops --create-namespace \
  -f k8s/helm/ids-api/values-prod.yaml \
  --set image.tag=$(git rev-parse --short HEAD)

# Désinstaller
helm uninstall ids-api -n securedmlops
```

### GitOps avec ArgoCD

```bash
./deploy-argocd.sh

# Accéder à l'UI ArgoCD
kubectl port-forward svc/argocd-server -n argocd 8080:443
kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath="{.data.password}" | base64 -d
```

→ Ouvrir https://localhost:8080 (user: `admin`)

### Gérer un Canary

```bash
# Déclencher une release
kubectl argo rollouts set image ids-api \
  ids-api=ghcr.io/oussamaelmoussaoui/ids-api:1.1.0 \
  -n securedmlops

# Suivre en live
kubectl argo rollouts get rollout ids-api -n securedmlops --watch

# Promouvoir l'étape suivante
kubectl argo rollouts promote ids-api -n securedmlops

# Rollback immédiat
kubectl argo rollouts abort ids-api -n securedmlops
kubectl argo rollouts undo ids-api -n securedmlops
```

### Sécurité Kubernetes (6 niveaux)

| Niveau | Mécanisme |
|--------|-----------|
| Cluster | Pod Security Standards `restricted` — pas de root, pas de conteneurs privilégiés |
| Pod | `readOnlyRootFilesystem`, `allowPrivilegeEscalation: false`, `capabilities: drop ALL` |
| Réseau | NetworkPolicies — Default DENY all, whitelist explicite uniquement |
| RBAC | ServiceAccount dédié, `automountServiceAccountToken: false` |
| Secrets | Aucune valeur en clair dans Git, HashiCorp Vault en production |
| Ingress | TLS obligatoire, rate limiting 100 req/min/IP, headers sécurité, WAF ModSecurity |

---

## 📈 Monitoring — Phase 6

### Architecture du monitoring

```
Vos données (reference.csv + current.csv)
          ↓
drift_metrics_exporter.py  :8001/metrics   (Evidently + Prometheus client)
          ↓
Prometheus                 :9090            (scraping toutes les 30s)
          ↓
Grafana                    :3000            (dashboards provisionnés)
```

### Pourquoi le drift monitoring ?

Quand un modèle ML est déployé, il a été entraîné sur des données d'une certaine période. Avec le temps, les données réelles changent (comportements, prix, tendances) — c'est le **data drift**. Le modèle continue de prédire silencieusement, mais de moins en moins bien. Sans monitoring, on découvre le problème seulement quand le business s'en plaint.

**Evidently** compare en continu les données d'entraînement (référence) et les données de production pour détecter ce glissement avant que les prédictions ne se dégradent.

### Configuration de l'environnement

Copier `.env.example` en `.env` à la racine du projet :

```cmd
copy .env.example .env
```

Contenu du fichier `.env` :
```env
GRAFANA_ADMIN_PASSWORD=your-secure-password

# Chemins vus par le conteneur (montés depuis DATA_ROOT)
REFERENCE_DATA_PATH=/data/reference.csv
CURRENT_DATA_PATH=/data/current.csv
TARGET_COLUMN=Label
DRIFT_INTERVAL_SECONDS=300

# Dossier host monté comme /data dans le conteneur
DATA_ROOT=../Model/data
```

### Démarrer la stack de monitoring

```cmd
cd monitoring
docker-compose up -d
```

### Accéder aux services

| Service | URL | Identifiants |
|---------|-----|-------------|
| Grafana | http://localhost:3000 | admin / `GRAFANA_ADMIN_PASSWORD` |
| Prometheus | http://localhost:9090 | — |
| Exporter métriques | http://localhost:8001/metrics | — |

### Smoke test

```cmd
curl http://localhost:8001/metrics
```

Sortie attendue :
```
data_drift_score 0.12
n_drifted_features 3.0
dataset_drift_detected 0.0
prediction_drift_score 0.04
feature_drift_score{feature="Flow Duration"} 0.08
```

### Générer un rapport HTML standalone (sans Docker)

```cmd
set REFERENCE_DATA_PATH=C:\chemin\vers\reference.csv
set CURRENT_DATA_PATH=C:\chemin\vers\current.csv
python monitoring/evidently/drift_report.py
```

Le rapport HTML et un fichier JSON de métriques sont écrits dans `monitoring/reports/`.

### Panels du dashboard Grafana

| Panel | Métrique Prometheus | Comment interpréter |
|-------|---------------------|---------------------|
| Overall Drift Score (gauge) | `data_drift_score` | Part des features driftées. > 0.15 → investiguer les données d'entrée |
| Drifted Features (stat) | `n_drifted_features` | Nombre brut de colonnes driftées |
| Dataset Drift Detected | `dataset_drift_detected` | 1 = drift global confirmé. 0 = stable |
| Prediction Drift Score | `prediction_drift_score` | Drift sur la colonne Label. > 0.10 → possible dégradation du modèle |
| Drift Over Time (time series) | les deux scores | Vue tendance — détecter une dégradation graduelle sur des heures/jours |
| Per-Feature Drift (barres) | `feature_drift_score` | Quelles features précises ont drifté (rouge = > 0.30) |

### Mettre à jour les données courantes

Pour pointer l'exporteur sur un nouveau snapshot de production :

```cmd
# Modifier CURRENT_DATA_PATH dans .env, puis :
cd monitoring
docker-compose restart evidently_exporter
```

### Arrêter la stack

```cmd
cd monitoring
docker-compose down
```

---

## 📁 Structure du Projet

```
SecuredMLOps/
│
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
│
├── k8s/
│   ├── base/                   ← Manifests Kubernetes bruts
│   ├── helm/ids-api/           ← Helm Chart complet
│   │   ├── templates/          ← rollout, services, ingress, HPA, PDB...
│   │   ├── values.yaml         ← Valeurs par défaut
│   │   ├── values-dev.yaml     ← Override Minikube
│   │   └── values-prod.yaml    ← Override Cloud
│   ├── rollouts/               ← Argo Rollouts (stratégie Canary)
│   ├── argocd/                 ← GitOps (AppProject + Application)
│   ├── security/               ← NetworkPolicies Zero Trust
│   ├── monitoring/             ← Prometheus rules SLO
│   └── scripts/
│       ├── setup-minikube.sh   ← Bootstrap complet
│       ├── deploy-api.sh       ← Build + déploiement Helm
│       ├── deploy-argocd.sh    ← Activer GitOps
│       └── simulate-canary.sh  ← Tester une release Canary
│
├── monitoring/
│   ├── evidently/
│   │   ├── drift_report.py         ← Rapport HTML standalone
│   │   └── drift_metrics_exporter.py ← Exporteur Prometheus :8001
│   ├── docker-compose.yml          ← Stack Prometheus + Grafana + Exporter
│   ├── prometheus/
│   │   └── prometheus.yml          ← Config scraping
│   ├── grafana/
│   │   ├── provisioning/           ← Datasources + dashboards auto-provisionnés
│   │   └── dashboards/
│   │       └── drift_dashboard.json
│   └── reports/                    ← Rapports HTML générés (gitignorés)
│
├── .dvc/                       ← Configuration DVC
├── .github/workflows/          ← CI/CD GitHub Actions
├── .env.example                ← Template variables d'environnement
├── params.yaml                 ← Configuration centralisée
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## ❓ Problèmes Fréquents

### Pipeline ML

| Problème | Solution |
|----------|----------|
| `Activate.ps1` bloqué | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `ModuleNotFoundError` | Vérifier que `(venv)` est visible dans le terminal |
| `dvc pull` — timeout DNS | Problème réseau université. Réessayer depuis une autre connexion |
| `dvc pull` — fichiers manquants | `git pull origin main` puis `dvc pull --force` |
| `MLflow connection refused` | Lancer `mlflow ui --port 5000` dans un terminal séparé d'abord |
| `MemoryError` pendant `train.py` | Fermer les autres applications — 8 GB RAM minimum requis |
| Tests `test_saved_model` skipped | Normal — lancer `train.py` d'abord pour générer le modèle |
| API `503 Service Unavailable` | Vérifier que `train.py` a bien tourné et généré le `.joblib` |

### Kubernetes

| Problème | Solution |
|----------|----------|
| `ImagePullBackOff` | Vérifier `eval $(minikube docker-env)` avant `docker build` |
| `CrashLoopBackOff` | `kubectl logs -n securedmlops -l app=ids-api --tail=100` |
| Pod `Pending` | `resources.requests` trop élevées — `kubectl describe pod` |
| `Readiness probe failed` | Fichiers `.joblib` absents de l'image — vérifier le Dockerfile |
| Ingress 404 | `minikube addons enable ingress` + entrée dans `/etc/hosts` |
| ArgoCD `OutOfSync` permanent | `argocd app diff ids-api` pour voir le diff |
| Rollout bloqué | `kubectl argo rollouts get rollout ids-api -n securedmlops` |
| NetworkPolicy bloque le trafic | `kubectl delete networkpolicy --all -n securedmlops` (temporaire) |

### Monitoring

| Problème | Solution |
|----------|----------|
| Toutes les métriques à 0 | `REFERENCE_DATA_PATH` / `CURRENT_DATA_PATH` inaccessibles dans le conteneur — vérifier `.env` et le mount `DATA_ROOT` |
| Dashboard non visible | Attendre ~30s après le démarrage pour le provisionnement Grafana, puis rafraîchir le navigateur |
| `ColumnDriftMetric` error | Les CSV référence et courant doivent avoir exactement les mêmes noms de colonnes |
| Grafana login échoue | `GRAFANA_ADMIN_PASSWORD` doit être défini dans `.env` **avant** le premier `docker-compose up` |

### Logs utiles

```bash
# API
kubectl logs -n securedmlops -l app=ids-api --tail=100 -f

# ArgoCD
kubectl logs -n argocd -l app.kubernetes.io/name=argocd-server

# Argo Rollouts
kubectl logs -n argo-rollouts -l app.kubernetes.io/name=argo-rollouts

# Ingress NGINX
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx
```

---

## ✅ Critères de Validation

### Phase 3 — ML
| Critère | Validation |
|---------|------------|
| Modèle entraîné | `Model/models/saved/xgboost_ids.joblib` existe |
| Performance | Accuracy > 97%, F1 > 0.97 |
| Tests | `pytest Model/tests/ -v` → 33 passed |
| API opérationnelle | `curl http://localhost:8000/health` → `200 OK` |

### Phase 5 — Kubernetes
| Critère | Validation |
|---------|------------|
| Déploiement Helm | `helm list -n securedmlops` → `ids-api deployed` |
| Pods running | `kubectl get pods -n securedmlops` → 3 pods `Running` |
| Canary fonctionnel | `simulate-canary.sh` progresse step par step |
| Rollback automatique | Injecter une erreur → rollback < 2 min |
| ArgoCD sync | UI ArgoCD → `Synced` + `Healthy` |
| Pas de root | `kubectl exec ... -- id` → `uid=1000` |

### Phase 6 — Monitoring
| Critère | Validation |
|---------|------------|
| Stack démarrée | `docker-compose up -d` sans erreur |
| Métriques exposées | `curl http://localhost:8001/metrics` → métriques drift visibles |
| Dashboard Grafana | http://localhost:3000 → dashboard `drift_dashboard` avec données live |
| Rapport HTML | `python monitoring/evidently/drift_report.py` → fichier dans `monitoring/reports/` |

---


**Liens du projet :**
- 🐙 GitHub : [oussamaelmoussaoui/SecuredMLOps](https://github.com/oussamaelmoussaoui/SecuredMLOps)
- 📊 DagsHub : [oussamaelmoussaoui/SecuredMLOps](https://dagshub.com/oussamaelmoussaoui/SecuredMLOps)
- 📖 Dataset : [CICIDS2017 — University of New Brunswick](https://www.unb.ca/cic/datasets/ids-2017.html)

---

<div align="center">
<i>Projet de Fin d'Année — Filière Data Science & Cloud Computing</i>
<br>
<i>SecuredMLOps — Pipeline MLOps Sécurisé · DevSecOps · Kubernetes · Drift Monitoring</i>
</div>
