#!/bin/bash
# ─────────────────────────────────────────
#  simulate-canary.sh
#  Simuler une release Canary
# ─────────────────────────────────────────
set -e

NAMESPACE="securedmlops"
NEW_TAG="${1:-1.1.0}"

echo "🐤 Simulation d'une release Canary"
echo "   Nouvelle version : ${NEW_TAG}"
echo ""

echo "📝 Mise à jour de l'image dans le Rollout..."
kubectl argo rollouts set image ids-api \
  ids-api=ghcr.io/oussamaelmoussaoui/ids-api:${NEW_TAG} \
  -n ${NAMESPACE}

echo "👀 Suivre la progression du Canary :"
echo ""
echo "   kubectl argo rollouts get rollout ids-api -n ${NAMESPACE} --watch"
echo ""
echo "🚦 Actions manuelles disponibles :"
echo "   - Promouvoir la prochaine étape :"
echo "     kubectl argo rollouts promote ids-api -n ${NAMESPACE}"
echo ""
echo "   - Faire un rollback :"
echo "     kubectl argo rollouts abort ids-api -n ${NAMESPACE}"
echo "     kubectl argo rollouts undo ids-api -n ${NAMESPACE}"
echo ""
echo "   - Pause / Resume :"
echo "     kubectl argo rollouts pause ids-api -n ${NAMESPACE}"
echo "     kubectl argo rollouts retry rollout ids-api -n ${NAMESPACE}"
