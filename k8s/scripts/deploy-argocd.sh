#!/bin/bash
# ─────────────────────────────────────────
#  deploy-argocd.sh
#  Déploiement via ArgoCD (GitOps)
# ─────────────────────────────────────────
set -e

echo "📦 [1/2] Création du projet ArgoCD..."
kubectl apply -f ../argocd/project.yaml

echo "🚀 [2/2] Création de l'Application ArgoCD..."
kubectl apply -f ../argocd/application.yaml

echo ""
echo "════════════════════════════════════════"
echo "✅ APPLICATION ARGOCD CRÉÉE"
echo "════════════════════════════════════════"
echo ""
echo "ArgoCD va maintenant surveiller le repo Git en"
echo "permanence. Tout commit sur la branche 'main'"
echo "déclenchera un déploiement automatique."
echo ""
echo "🔗 Suivre dans l'UI ArgoCD :"
echo "   kubectl port-forward svc/argocd-server -n argocd 8080:443"
echo "   → https://localhost:8080"
echo ""
echo "🔗 Suivre via CLI :"
echo "   kubectl get applications -n argocd -w"
