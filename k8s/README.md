# 🚢 Phase 5 — Déploiement Kubernetes avec Canary & GitOps

> **Objectif :** déployer l'IDS API sur Kubernetes avec une stratégie de release progressive (**Canary**), gérée en **GitOps** via ArgoCD, en respectant l'approche **DevSecOps** du projet.

---

## 📋 Table des matières

- [Architecture Phase 5](#-architecture-phase-5)
- [Stack technique](#-stack-technique)
- [Structure des fichiers](#-structure-des-fichiers)
- [Prérequis](#-prérequis)
- [Installation locale (Minikube)](#-installation-locale-minikube)
- [Déploiement Helm](#-déploiement-helm)
- [GitOps avec ArgoCD](#-gitops-avec-argocd)
- [Canary Deployment avec Argo Rollouts](#-canary-deployment-avec-argo-rollouts)
- [Sécurité (DevSecOps)](#-sécurité-devsecops)
- [Déploiement Cloud (AKS/EKS/GKE)](#-déploiement-cloud-akseksgke)
- [Troubleshooting](#-troubleshooting)

---

## 🏗 Architecture Phase 5

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
              ▲                                       ▲
              │                                       │
              └────────── Service Stable ─────────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │   Ingress NGINX      │
                       │  + TLS + Rate Limit  │
                       └──────────┬───────────┘
                                  │
                                  ▼
                            Utilisateurs
```

**Flux Canary :**
```
10% → pause 2min → analyse Prometheus ─┐
25% → pause 2min → analyse Prometheus ─┤  ❌ Échec → rollback auto
50% → pause 5min → analyse Prometheus ─┤
75% → pause 2min → analyse Prometheus ─┘
100% → version stable promue
```

---

## 🛠 Stack technique

| Outil | Rôle |
|-------|------|
| **Kubernetes** | Orchestration des conteneurs |
| **Helm** | Packaging des manifests |
| **Argo Rollouts** | Stratégie Canary + rollback automatique |
| **ArgoCD** | GitOps — synchronisation Git ↔ cluster |
| **NGINX Ingress** | Routage HTTP/HTTPS + rate limiting + WAF |
| **Prometheus** | Métriques pour les analyses Canary |
| **cert-manager** | TLS automatique (Let's Encrypt) en cloud |

---

## 📁 Structure des fichiers

```
k8s/
├── README.md                         ← Ce fichier
│
├── base/                             ← Manifests bruts (sans Helm)
│   ├── 00-namespace.yaml             ← Namespace + Pod Security Standards
│   ├── 01-configmap.yaml             ← Config non sensible
│   ├── 02-secret.yaml                ← Template secrets
│   ├── 03-serviceaccount.yaml        ← SA + RBAC moindre privilège
│   ├── 04-deployment.yaml            ← Deployment hardened
│   ├── 05-service.yaml               ← Service ClusterIP
│   ├── 06-ingress.yaml               ← Ingress TLS + rate-limit
│   ├── 07-hpa.yaml                   ← Autoscaling 3→10 pods
│   └── 08-pdb.yaml                   ← PodDisruptionBudget
│
├── helm/ids-api/                     ← Helm Chart
│   ├── Chart.yaml
│   ├── values.yaml                   ← Valeurs par défaut
│   ├── values-dev.yaml               ← Override dev (Minikube)
│   ├── values-prod.yaml              ← Override prod (Cloud)
│   └── templates/
│       ├── _helpers.tpl
│       ├── configmap.yaml
│       ├── secret.yaml
│       ├── serviceaccount.yaml
│       ├── rollout.yaml              ← Rollout OU Deployment selon strategy
│       ├── service.yaml              ← Stable + Canary services
│       ├── ingress.yaml
│       ├── hpa.yaml
│       ├── pdb.yaml
│       ├── analysis-templates.yaml   ← Vérifs Prometheus pour Canary
│       ├── networkpolicy.yaml
│       └── servicemonitor.yaml       ← Prometheus Operator (Phase 6)
│
├── rollouts/                         ← Argo Rollouts (sans Helm)
│   ├── rollout.yaml                  ← Stratégie Canary complète
│   ├── services.yaml                 ← Stable + Canary
│   └── analysis-templates.yaml       ← success-rate + latency-p95
│
├── argocd/                           ← GitOps
│   ├── project.yaml                  ← AppProject restreint
│   └── application.yaml              ← Application surveillée
│
├── security/                         ← DevSecOps
│   └── network-policies.yaml         ← Zero Trust networking
│
├── monitoring/                       ← Phase 6
│   └── prometheus-rules.yaml         ← Alertes SLO
│
└── scripts/                          ← Automatisation
    ├── setup-minikube.sh             ← Bootstrap complet
    ├── deploy-api.sh                 ← Build + déploiement Helm
    ├── deploy-argocd.sh              ← Activer GitOps
    └── simulate-canary.sh            ← Tester une release Canary
```

---

## ⚙️ Prérequis

| Outil | Version min | Installation |
|-------|-------------|--------------|
| Docker Desktop | 24.x | [docker.com](https://www.docker.com/products/docker-desktop/) |
| kubectl | 1.29+ | [kubernetes.io](https://kubernetes.io/docs/tasks/tools/) |
| Minikube | 1.32+ | [minikube.sigs.k8s.io](https://minikube.sigs.k8s.io/) |
| Helm | 3.13+ | [helm.sh](https://helm.sh/docs/intro/install/) |
| kubectl-argo-rollouts | 1.6+ | [argoproj/argo-rollouts](https://argoproj.github.io/argo-rollouts/installation/#kubectl-plugin-installation) |

```bash
# Vérifier les versions
docker --version
kubectl version --client
minikube version
helm version
kubectl argo rollouts version
```

---

## 🚀 Installation locale (Minikube)

### Étape 1 — Bootstrap de l'environnement

```bash
cd k8s/scripts
chmod +x *.sh
./setup-minikube.sh
```

Ce script installe :
- Minikube (4 CPU, 8 GB RAM)
- NGINX Ingress Controller
- Metrics Server (pour HPA)
- Argo Rollouts (namespace `argo-rollouts`)
- ArgoCD (namespace `argocd`)
- Le namespace `securedmlops`

> ⏳ 5 à 10 minutes selon ta connexion.

### Étape 2 — Build de l'image dans Minikube

```bash
./deploy-api.sh
```

Ce script :
1. Build l'image Docker directement dans le daemon Docker de Minikube (pas besoin de registry)
2. Crée les secrets K8s à partir de variables d'env
3. Déploie via Helm avec `values-dev.yaml`
4. Affiche le statut du cluster

### Étape 3 — Tester l'API

```bash
# Port-forward simple
kubectl port-forward -n securedmlops svc/ids-api 8000:80

# Dans un autre terminal
curl http://localhost:8000/health
curl http://localhost:8000/model/info
```

**Ou via Ingress :**

```bash
# Récupérer l'IP de Minikube
minikube ip

# Ajouter dans /etc/hosts (Linux/Mac) ou C:\Windows\System32\drivers\etc\hosts
# <minikube-ip>  ids-api.local

# Tester
curl http://ids-api.local/docs
```

---

## ⛵ Déploiement Helm

### Sans GitOps (déploiement manuel)

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
```

### Override d'une valeur ponctuelle

```bash
helm upgrade ids-api k8s/helm/ids-api \
  -n securedmlops \
  --set replicaCount=5 \
  --reuse-values
```

### Désinstaller

```bash
helm uninstall ids-api -n securedmlops
```

---

## 🔄 GitOps avec ArgoCD

### Principe

ArgoCD compare en permanence l'état désiré (le code dans Git) avec l'état réel du cluster. Tout écart est automatiquement corrigé. **Le Git devient la seule source de vérité.**

### Activation

```bash
cd k8s/scripts
./deploy-argocd.sh
```

### Accéder à l'UI ArgoCD

```bash
# Port-forward
kubectl port-forward svc/argocd-server -n argocd 8080:443

# Récupérer le mot de passe admin
kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath="{.data.password}" | base64 -d
```

→ Ouvrir [https://localhost:8080](https://localhost:8080) (user: `admin`)

### Workflow GitOps

```
1. Développeur push un commit dans main
        ↓
2. ArgoCD détecte le changement (polling 3 min ou webhook)
        ↓
3. ArgoCD diff avec le cluster
        ↓
4. ArgoCD applique les changements
        ↓
5. Argo Rollouts orchestre le Canary
        ↓
6. Promotion automatique si SLO respectés
```

### Commandes utiles

```bash
# Lister les applications
kubectl get applications -n argocd

# Forcer une sync
kubectl patch app ids-api -n argocd \
  --type merge \
  --patch '{"operation":{"sync":{}}}'

# Voir l'historique de sync
argocd app history ids-api
```

---

## 🐤 Canary Deployment avec Argo Rollouts

### Stratégie configurée

| Step | % trafic | Pause | Vérification |
|------|----------|-------|--------------|
| 1 | 10% | 2 min | success-rate ≥ 95% + latence p95 ≤ 100ms |
| 2 | 25% | 2 min | idem |
| 3 | 50% | 5 min | idem |
| 4 | 75% | 2 min | idem |
| 5 | 100% | — | promotion finale |

### Déclencher une release Canary

```bash
# Méthode 1 : push d'un nouveau tag dans Git → ArgoCD déclenche
git tag v1.1.0 && git push origin v1.1.0

# Méthode 2 : mise à jour manuelle
kubectl argo rollouts set image ids-api \
  ids-api=ghcr.io/oussamaelmoussaoui/ids-api:1.1.0 \
  -n securedmlops

# Ou via le script
./scripts/simulate-canary.sh 1.1.0
```

### Suivre la progression

```bash
# Suivi en live
kubectl argo rollouts get rollout ids-api -n securedmlops --watch

# Dashboard graphique
kubectl argo rollouts dashboard
# → http://localhost:3100
```

### Actions manuelles

```bash
# Promouvoir l'étape suivante (skip la pause)
kubectl argo rollouts promote ids-api -n securedmlops

# Pause indéfinie
kubectl argo rollouts pause ids-api -n securedmlops

# Reprendre
kubectl argo rollouts retry rollout ids-api -n securedmlops

# Rollback immédiat
kubectl argo rollouts abort ids-api -n securedmlops
kubectl argo rollouts undo ids-api -n securedmlops
```

### Rollback automatique

Si Prometheus détecte :
- success-rate < 95% pendant 2 mesures consécutives, **OU**
- latence p95 > 100ms pendant 2 mesures consécutives

→ **Argo Rollouts rollback automatiquement** vers la version stable.

---

## 🛡️ Sécurité (DevSecOps)

Cette phase intègre la sécurité à **6 niveaux** :

### 1️⃣ Niveau Cluster — Pod Security Standards

Le namespace `securedmlops` impose le profil **`restricted`** :
- Pas de conteneurs privilégiés
- Pas de root
- Pas de hostPath / hostNetwork
- seccompProfile obligatoire

### 2️⃣ Niveau Pod — SecurityContext

```yaml
runAsNonRoot: true
runAsUser: 1000
readOnlyRootFilesystem: true
allowPrivilegeEscalation: false
capabilities:
  drop: [ALL]
seccompProfile:
  type: RuntimeDefault
```

### 3️⃣ Niveau Réseau — NetworkPolicies

- Default DENY all (ingress + egress)
- Whitelist explicite uniquement
- Trafic autorisé : Ingress NGINX, DNS, MLflow, HTTPS sortant

### 4️⃣ Niveau RBAC — Moindre privilège

- ServiceAccount dédié
- `automountServiceAccountToken: false`
- Role limité à `configmaps:get/list/watch`

### 5️⃣ Niveau Secrets

- Aucune valeur en clair dans Git
- Création via `kubectl create secret` ou Vault
- En production : **HashiCorp Vault + External Secrets Operator** (Phase 4)
- Alternative : Sealed Secrets de Bitnami

### 6️⃣ Niveau Ingress — Couche applicative

- TLS obligatoire (HSTS + force-ssl-redirect)
- Rate limiting (100 req/min/IP)
- Headers de sécurité (X-Frame-Options, CSP, etc.)
- WAF ModSecurity + règles OWASP CRS (production)

---

## ☁️ Déploiement Cloud (AKS/EKS/GKE)

### Différences avec Minikube

| Aspect | Minikube | Cloud |
|--------|----------|-------|
| Image | Local Docker | Registry (GHCR / ACR / ECR / GCR) |
| TLS | Auto-signé (désactivé) | cert-manager + Let's Encrypt |
| Stockage | hostPath | Persistent Volumes (Azure Disk, EBS, GCE PD) |
| LoadBalancer | NodePort | Cloud LB natif |
| Secrets | Manuel | Vault / Cloud KMS |
| Observabilité | Optionnel | Stack complète (Phase 6) |

### Exemple — AKS (Azure Kubernetes Service)

```bash
# 1. Créer le cluster
az aks create \
  --resource-group rg-securedmlops \
  --name aks-securedmlops \
  --node-count 3 \
  --enable-managed-identity \
  --enable-addons monitoring \
  --network-plugin azure \
  --network-policy calico

# 2. Configurer kubectl
az aks get-credentials \
  --resource-group rg-securedmlops \
  --name aks-securedmlops

# 3. Push de l'image vers ACR
az acr login --name acrSecuredMLOps
docker tag ids-api:1.0.0 acrSecuredMLOps.azurecr.io/ids-api:1.0.0
docker push acrSecuredMLOps.azurecr.io/ids-api:1.0.0

# 4. Déployer
helm upgrade --install ids-api k8s/helm/ids-api \
  -n securedmlops --create-namespace \
  -f k8s/helm/ids-api/values-prod.yaml \
  --set image.repository=acrSecuredMLOps.azurecr.io/ids-api
```

---

## 🩺 Troubleshooting

| Problème | Solution |
|----------|----------|
| `ImagePullBackOff` | Vérifier `eval $(minikube docker-env)` avant `docker build` |
| `CrashLoopBackOff` | `kubectl logs -n securedmlops -l app=ids-api --tail=100` |
| Pod `Pending` | Probablement les `resources.requests` trop élevées — `kubectl describe pod` |
| `Readiness probe failed` | Modèle pas chargé. Vérifier que les fichiers `.joblib` sont dans l'image |
| Ingress 404 | Vérifier `minikube addons enable ingress` + entry dans `/etc/hosts` |
| ArgoCD `OutOfSync` permanent | Diff manuel : `argocd app diff ids-api` |
| Rollout bloqué | `kubectl argo rollouts get rollout ids-api -n securedmlops` pour voir l'étape |
| NetworkPolicy bloque le trafic | Désactiver temporairement : `kubectl delete networkpolicy --all -n securedmlops` |

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

## ✅ Critères de validation Phase 5

| Critère | Validation |
|---------|------------|
| Déploiement Helm fonctionnel | `helm list -n securedmlops` montre `ids-api deployed` |
| Pods running | `kubectl get pods -n securedmlops` → 3 pods `Running` |
| API accessible | `curl http://ids-api.local/health` → `200 OK` |
| Canary fonctionnel | `simulate-canary.sh` progresse step par step |
| Rollback automatique | Injecter une erreur → rollback < 2 min |
| ArgoCD sync OK | UI ArgoCD → `Synced` + `Healthy` |
| HPA fonctionnel | Load test → scale up automatique |
| NetworkPolicies actives | `kubectl get networkpolicy -n securedmlops` |
| Pas de root dans les pods | `kubectl exec ... -- id` → `uid=1000` |

---

<div align="center">
<i>Phase 5 — Déploiement Kubernetes + GitOps + Canary</i>
<br>
<i>SecuredMLOps — Pipeline MLOps Sécurisé</i>
</div>
