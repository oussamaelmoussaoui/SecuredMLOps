#!/bin/bash
# ─────────────────────────────────────────
#  deploy-api.sh
#  Build de l'image dans Minikube + déploiement Helm
# ─────────────────────────────────────────
set -e

NAMESPACE="securedmlops"
IMAGE_TAG="1.0.0-dev"

echo "🔨 [1/4] Build de l'image dans le daemon Docker de Minikube..."
eval $(minikube docker-env)

cd ../..   # racine du repo
docker build -t ghcr.io/oussamaelmoussaoui/ids-api:${IMAGE_TAG} .
cd k8s/scripts

echo "🔐 [2/4] Création des secrets..."
# ⚠️ Adapter avec tes vraies valeurs ou utiliser un .env
kubectl create secret generic ids-api-secrets \
  --namespace=${NAMESPACE} \
  --from-literal=DAGSHUB_USER="${DAGSHUB_USER:-placeholder}" \
  --from-literal=DAGSHUB_TOKEN="${DAGSHUB_TOKEN:-placeholder}" \
  --from-literal=MLFLOW_TRACKING_USERNAME="${MLFLOW_USER:-placeholder}" \
  --from-literal=MLFLOW_TRACKING_PASSWORD="${MLFLOW_PWD:-placeholder}" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "⛵ [3/4] Déploiement via Helm (values-dev)..."
helm upgrade --install ids-api ../helm/ids-api \
  --namespace=${NAMESPACE} \
  --create-namespace \
  -f ../helm/ids-api/values-dev.yaml \
  --wait \
  --timeout=5m

echo "🔎 [4/4] Statut du déploiement..."
kubectl get all -n ${NAMESPACE}

echo ""
echo "════════════════════════════════════════"
echo "✅ DÉPLOIEMENT RÉUSSI"
echo "════════════════════════════════════════"
echo ""
echo "🔗 Tester l'API :"
echo "   kubectl port-forward -n ${NAMESPACE} svc/ids-api 8000:80"
echo "   curl http://localhost:8000/health"
echo ""
echo "🔗 Accès via Ingress :"
echo "   Ajouter dans /etc/hosts (ou Windows hosts) :"
echo "   \$(minikube ip)  ids-api.local"
echo "   → http://ids-api.local/docs"
