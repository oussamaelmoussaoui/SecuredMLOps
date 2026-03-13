# ─────────────────────────────────────────
#  Dockerfile
#  Image Docker — API de Détection d'Intrusions
# ─────────────────────────────────────────

FROM python:3.10-slim

# Métadonnées
LABEL maintainer="SecuredMLOps"
LABEL description="IDS API — XGBoost CICIDS2017"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# Dossier de travail
WORKDIR $APP_HOME

# Installer les dépendances système (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY Model/ ./Model/
COPY params.yaml .

# Créer les dossiers nécessaires
RUN mkdir -p Model/data/processed Model/models/saved Model/docs

# Utilisateur non-root pour la sécurité
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser $APP_HOME
USER appuser

# Exposer le port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Commande de démarrage
CMD ["uvicorn", "Model.src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
