#!/bin/bash
# ─────────────────────────────────────────
#  setup-minikube.sh
#  Bootstrap complet de l'environnement local
#  Minikube + Ingress + ArgoCD + Argo Rollouts
# ─────────────────────────────────────────
set -e

echo "🚀 [1/7] Démarrage de Minikube..."
minikube start \
  --cpus=4 \
  --memory=8192 \
  --driver=docker \
  --kubernetes-version=v1.29.0

echo "🌐 [2/7] Activation de l'Ingress NGINX..."
minikube addons enable ingress
minikube addons enable metrics-server

echo "⏳ Attente du démarrage de l'ingress-nginx..."
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=180s

echo "📦 [3/7] Installation d'Argo Rollouts..."
kubectl create namespace argo-rollouts --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -n argo-rollouts \
  -f https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml

echo "🔄 [4/7] Installation d'ArgoCD..."
kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -n argocd \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

echo "⏳ Attente du démarrage d'ArgoCD..."
kubectl wait --namespace argocd \
  --for=condition=available --timeout=300s \
  deployment/argocd-server

echo "🛡️  [5/7] Création du namespace projet..."
kubectl apply -f ../base/00-namespace.yaml

echo "🔑 [6/7] Récupération du mot de passe ArgoCD..."
ARGOCD_PWD=$(kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath="{.data.password}" | base64 -d)

echo ""
echo "════════════════════════════════════════"
echo "✅ ENVIRONNEMENT PRÊT"
echo "════════════════════════════════════════"
echo ""
echo "🔗 ArgoCD UI :"
echo "   kubectl port-forward svc/argocd-server -n argocd 8080:443"
echo "   → https://localhost:8080"
echo "   user: admin"
echo "   pass: $ARGOCD_PWD"
echo ""
echo "🔗 Argo Rollouts dashboard :"
echo "   kubectl argo rollouts dashboard"
echo "   → http://localhost:3100"
echo ""
echo "📌 [7/7] Étape suivante :"
echo "   bash deploy-api.sh"
