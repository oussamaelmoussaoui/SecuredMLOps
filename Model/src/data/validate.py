# ─────────────────────────────────────────
#  Model/src/data/validate.py
#  Étape 3 — Validation de qualité des données
# ─────────────────────────────────────────
#
#  COMMENT UTILISER :
#  > python Model/src/data/validate.py
#
#  INPUT  : Model/data/raw/*.csv
#  OUTPUT : Model/data/validation_report.html  (rapport visuel)
#           Console : ✅ ou ❌ pour chaque règle
# ─────────────────────────────────────────

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR      = PROJECT_ROOT / "Model" / "data" / "raw"
REPORT_DIR   = PROJECT_ROOT / "Model" / "data"
PARAMS_FILE  = PROJECT_ROOT / "params.yaml"


def load_sample(raw_dir: Path, nrows: int = 50000) -> pd.DataFrame:
    """Charger un échantillon rapide pour la validation."""
    files = list(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Aucun CSV dans {raw_dir}")

    dfs = []
    rows_per_file = max(nrows // len(files), 1000)
    for f in files:
        df = pd.read_csv(f, nrows=rows_per_file, low_memory=False)
        df.columns = df.columns.str.strip()
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# ─────────────────────────────────────────
#  RÈGLES DE VALIDATION CICIDS2017
# ─────────────────────────────────────────
def run_validation(df: pd.DataFrame) -> dict:
    """
    Exécuter toutes les règles de validation.
    Retourne un dict avec les résultats de chaque règle.
    """
    results = {}

    # ── Règle 1 : Colonnes essentielles présentes ──
    required_cols = [
        "Flow Duration", "Total Fwd Packets",
        "Total Backward Packets", "Label"
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    results["R1_colonnes_essentielles"] = {
        "passed": len(missing_cols) == 0,
        "detail": f"Manquantes: {missing_cols}" if missing_cols else f"{len(required_cols)} colonnes OK"
    }

    # ── Règle 2 : Label présent et non vide ──
    if "Label" in df.columns:
        null_labels = df["Label"].isnull().sum()
        results["R2_label_non_null"] = {
            "passed": null_labels == 0,
            "detail": f"{null_labels} labels manquants"
        }

        # ── Règle 3 : Labels valides CICIDS2017 ──
        valid_labels = {
            "BENIGN", "DoS Hulk", "PortScan", "DDoS",
            "DoS GoldenEye", "DoS Slowloris", "DoS Slowhttptest",
            "FTP-Patator", "SSH-Patator", "Web Attack – Brute Force",
            "Web Attack – XSS", "Web Attack – Sql Injection",
            "Infiltration", "Bot", "Heartbleed"
        }
        unknown_labels = set(df["Label"].dropna().unique()) - valid_labels
        results["R3_labels_valides"] = {
            "passed": len(unknown_labels) == 0,
            "detail": f"Labels inconnus: {unknown_labels}" if unknown_labels
                      else f"{df['Label'].nunique()} classes reconnues"
        }

    # ── Règle 4 : Pas de valeurs infinies ──
    numeric_df = df.select_dtypes(include=[np.number])
    inf_count  = np.isinf(numeric_df.values).sum()
    results["R4_pas_de_inf"] = {
        "passed": inf_count == 0,
        "detail": f"{inf_count} valeurs infinies trouvées"
    }

    # ── Règle 5 : Taux de NaN acceptable (< 5%) ──
    nan_rate = df.isnull().mean().mean()
    results["R5_taux_nan"] = {
        "passed": nan_rate < 0.05,
        "detail": f"Taux NaN global : {nan_rate:.2%}"
    }

    # ── Règle 6 : Flow Duration positif ──
    if "Flow Duration" in df.columns:
        neg_duration = (df["Flow Duration"] < 0).sum()
        results["R6_flow_duration_positif"] = {
            "passed": neg_duration == 0,
            "detail": f"{neg_duration} durées négatives"
        }

    # ── Règle 7 : Nombre de colonnes attendu (≥ 70) ──
    n_cols = df.shape[1]
    results["R7_nombre_colonnes"] = {
        "passed": n_cols >= 70,
        "detail": f"{n_cols} colonnes (minimum attendu : 70)"
    }

    # ── Règle 8 : Taille minimale du dataset ──
    n_rows = len(df)
    results["R8_taille_dataset"] = {
        "passed": n_rows >= 1000,
        "detail": f"{n_rows:,} lignes (minimum : 1 000)"
    }

    # ── Règle 9 : Déséquilibre de classes détecté ──
    if "Label" in df.columns:
        class_counts  = df["Label"].value_counts(normalize=True)
        benign_ratio  = class_counts.get("BENIGN", 0)
        results["R9_desequilibre_classes"] = {
            "passed": True,   # Info seulement, pas bloquant
            "detail": f"BENIGN={benign_ratio:.1%} — Déséquilibre {'⚠️ élevé' if benign_ratio > 0.7 else 'OK'}"
        }

    return results


# ─────────────────────────────────────────
#  RAPPORT HTML
# ─────────────────────────────────────────
def generate_html_report(results: dict, df: pd.DataFrame, output_path: Path):
    """Générer un rapport HTML de validation."""
    passed = sum(1 for r in results.values() if r["passed"])
    total  = len(results)
    status_color = "#2ecc71" if passed == total else "#e74c3c"

    rows_html = ""
    for rule_name, res in results.items():
        icon    = "✅" if res["passed"] else "❌"
        bg      = "#eafaf1" if res["passed"] else "#fdedec"
        rows_html += f"""
        <tr style="background:{bg}">
            <td>{icon}</td>
            <td><b>{rule_name}</b></td>
            <td>{res['detail']}</td>
        </tr>"""

    class_dist = ""
    if "Label" in df.columns:
        for label, count in df["Label"].value_counts().items():
            pct = count / len(df) * 100
            class_dist += f"<tr><td>{label}</td><td>{count:,}</td><td>{pct:.1f}%</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Rapport de Validation — CICIDS2017</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        .summary {{ background:{status_color}; color:white; padding:15px; border-radius:8px; margin:20px 0; }}
        table {{ width:100%; border-collapse:collapse; margin:20px 0; }}
        th {{ background:#2c3e50; color:white; padding:10px; text-align:left; }}
        td {{ padding:10px; border-bottom:1px solid #ddd; }}
    </style>
</head>
<body>
    <h1>🔍 Rapport de Validation — CICIDS2017</h1>
    <div class="summary">
        <h2>Résultat Global : {passed}/{total} règles passées</h2>
        <p>Dataset : {len(df):,} lignes | {df.shape[1]} colonnes</p>
    </div>

    <h2>Règles de Validation</h2>
    <table>
        <tr><th>Statut</th><th>Règle</th><th>Détail</th></tr>
        {rows_html}
    </table>

    <h2>Distribution des Classes</h2>
    <table>
        <tr><th>Classe</th><th>Count</th><th>Pourcentage</th></tr>
        {class_dist}
    </table>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Rapport HTML généré : {output_path}")


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
def main():
    logger.info("═══════════════════════════════════════")
    logger.info("  VALIDATION DES DONNÉES CICIDS2017")
    logger.info("═══════════════════════════════════════")

    logger.info("Chargement d'un échantillon...")
    df = load_sample(RAW_DIR)
    logger.info(f"Échantillon : {df.shape[0]:,} lignes")

    logger.info("\nExécution des règles de validation...")
    results = run_validation(df)

    # Affichage console
    print("\n" + "─"*55)
    print(f"  {'RÈGLE':<40} {'STATUT'}")
    print("─"*55)
    all_passed = True
    for rule, res in results.items():
        icon = "✅ PASS" if res["passed"] else "❌ FAIL"
        print(f"  {rule:<40} {icon}")
        print(f"    → {res['detail']}")
        if not res["passed"] and rule != "R9_desequilibre_classes":
            all_passed = False
    print("─"*55)

    # Rapport HTML
    report_path = REPORT_DIR / "validation_report.html"
    generate_html_report(results, df, report_path)

    if all_passed:
        logger.info("\n✅ Validation réussie ! Prochaine étape :")
        logger.info("   > python Model/src/data/preprocess.py")
    else:
        logger.error("\n❌ Validation échouée — Corrigez les erreurs avant de continuer.")
        exit(1)


if __name__ == "__main__":
    main()
