# Secrets GitHub Actions requis — SecuredMLOps

À configurer dans : **Settings → Secrets and variables → Actions → New repository secret**

## Secrets obligatoires

| Secret | Utilisé dans | Description |
|--------|-------------|-------------|
| `DAGSHUB_USERNAME` | `training.yml`, `docker-build-scan.yml` | Nom d'utilisateur DagsHub pour accès DVC remote |
| `DAGSHUB_TOKEN` | `training.yml`, `docker-build-scan.yml` | Token d'accès DagsHub (générer sur dagshub.com → Settings → Tokens) |
| `DOCKERHUB_USERNAME` | `docker-build-scan.yml`, `deploy-staging.yml`, `deploy-production.yml`, `rollback.yml` | Nom d'utilisateur Docker Hub |
| `DOCKERHUB_TOKEN` | `docker-build-scan.yml`, `deploy-staging.yml`, `deploy-production.yml`, `rollback.yml` | Token Docker Hub (générer sur hub.docker.com → Account Settings → Security) |

## Secret automatique (fourni par GitHub)

| Secret | Utilisé dans | Description |
|--------|-------------|-------------|
| `GITHUB_TOKEN` | `ci.yml` (GitLeaks) | Fourni automatiquement par GitHub Actions — aucune configuration requise |

## Comment créer les secrets

```bash
# Via GitHub CLI (gh)
gh secret set DAGSHUB_USERNAME --body "votre_username_dagshub"
gh secret set DAGSHUB_TOKEN    --body "votre_token_dagshub"
gh secret set DOCKERHUB_USERNAME --body "votre_username_dockerhub"
gh secret set DOCKERHUB_TOKEN    --body "votre_token_dockerhub"
```

## Environnements GitHub requis

Les workflows suivants utilisent des **environments** GitHub (Settings → Environments) :

| Environnement | Workflow | Utilité |
|---------------|---------|---------|
| `staging` | `deploy-staging.yml` | Approbation optionnelle avant staging |
| `production` | `deploy-production.yml`, `rollback.yml` | Approbation manuelle recommandée avant production |

> Pour activer les approbations : Settings → Environments → `production` → Required reviewers
