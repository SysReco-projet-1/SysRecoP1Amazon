import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ================================
# Détection de la racine du projet
# ================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = Path(__file__).resolve().parent

# Chemin vers les deux fichiers csv
file_path_50k = PROJECT_ROOT / "3.1.1" / "amazon_books_sample_active_users.csv"
file_path_temp = PROJECT_ROOT / "3.1.1" / "amazon_books_sample_temporal.csv"

# ============================================================================
# 3. Analyse de la distribution :
# — Distribution de la popularité des livres (phénomène de ≪ longue traı̂ne ≫)
# — Distribution temporelle des évaluations
# — Distribution des votes d’utilité (helpful vote)
# — Proportion d’achats vérifiés (verified purchase)
# ============================================================================

# Fonction effectuant la tâche 3
def task_analyse_distribution(file_path):

    # Nom pour différentier les fichier csv source
    output_name = file_path.stem

    # Charger le fichier CSV
    df = pd.read_csv(file_path)

    # Popularité des livres
    item_popularity = df.groupby("parent_asin").size()

    # Distribution temporelle
    if "year" in df.columns:
        temporal_distribution = df["year"].value_counts().sort_index()
    else:
        df["year"] = pd.to_datetime(df["timestamp"], unit="s").dt.year
        temporal_distribution = df["year"].value_counts().sort_index()

    # Votes utiles
    helpful_votes = df["helpful_vote"]

    # Achats vérifiés
    verified_counts = df["verified_purchase"].value_counts()
    verified_ratio = df["verified_purchase"].mean()

    # ===========================
    # Écriture dans fichier texte
    # ===========================
    output_txt = TASK_ROOT / f"{output_name}_distribution_analysis.txt"

    with open(output_txt, "w", encoding="utf-8") as f:

        def write(line=""):
            print(line)
            f.write(line + "\n")

        write("===== Distribution de la popularité des livres =====")
        write(f"Nombre de livres : {len(item_popularity)}")
        write(f"Popularité moyenne : {item_popularity.mean():.2f}")
        write(f"Popularité max : {item_popularity.max()}")
        write()

        write("===== Distribution temporelle des évaluations =====")
        write(temporal_distribution.to_string())
        write()

        write("===== Distribution des votes d’utilité =====")
        write(helpful_votes.describe().to_string())
        write()

        write("===== Achats vérifiés =====")
        write(verified_counts.to_string())
        write(f"Proportion d’achats vérifiés : {verified_ratio:.4f}")

    # ====================================================================
    # Distrubution de la popularité des livres avec un graphique Long Tail 
    # ====================================================================

    # Données pour le graphique
    popularity = df.groupby("parent_asin").size()
    sorted_popularity = popularity.sort_values(ascending=False).values

    x = np.arange(1, len(sorted_popularity) + 1)

    # Aire sous courbe
    area = np.trapz(sorted_popularity, x)

    # Distribution cumulée
    cumulative = np.cumsum(sorted_popularity)
    total = cumulative[-1]

    # Seuil 80%
    threshold = 0.8 * total
    head_index = np.searchsorted(cumulative, threshold)

    plt.figure(figsize=(10, 6))

    # Courbe
    plt.plot(x, sorted_popularity, linewidth=2)

    # Aire totale
    plt.fill_between(x, sorted_popularity, alpha=0.2)

    # Zone HEAD
    plt.fill_between(
        x[:head_index],
        sorted_popularity[:head_index],
        alpha=0.5,
        label="Head (80% interactions)"
    )

    # Ligne séparation
    plt.axvline(head_index, linestyle="--", linewidth=2)

    plt.title(
        f"Distribution de la popularité des livres (Long Tail)\n"
        f"Aire = {area:.2e} — {output_name}"
    )

    plt.xlabel("Livres triés par popularité")
    plt.ylabel("Nombre d’évaluations")

    plt.yscale("log")

    plt.legend()

    plt.savefig(TASK_ROOT / f"{output_name}_long_tail_head_tail.png",
                dpi=300, bbox_inches="tight")

    # =======================================
    # Distribution temporelle des évaluations
    # =======================================
    plt.figure(figsize=(8, 5))

    temporal_distribution.plot(kind="bar")

    plt.title(f"Distribution temporelle des évaluations — {output_name}")
    plt.xlabel("Année")
    plt.ylabel("Nombre d’évaluations")

    plt.savefig(TASK_ROOT / f"{output_name}_temporal_distribution.png", dpi=300, bbox_inches="tight")

    # ===================================================================
    # Proportions des évaluations marquées utiles par d'autres utilisateurs
    # ===================================================================

    # Regroupement des helpful votes en fourchette pour une meilleurs représentation de la proportion
    helpful_bins = pd.cut(
    df["helpful_vote"],
    bins=[-1, 0, 1, 5, 10, 50, 100, float("inf")],
    labels=["0", "1", "2-5", "6-10", "11-50", "51-100", "100+"]
    )

    helpful_counts = helpful_bins.value_counts().sort_index()

    plt.figure(figsize=(8, 5))

    helpful_counts.plot(kind="bar")

    plt.title(f"Distribution des évaluations marquées utiles — {output_name}")
    plt.xlabel("Nombre d’évaluations utiles")
    plt.ylabel("Nombre d’évaluations")

    plt.xticks(rotation=0)

    plt.savefig(TASK_ROOT / f"{output_name}_helpful_votes.png", dpi=300, bbox_inches="tight")


    # =================================================
    # Proportions des achats vérifiés des livre évalués
    # =================================================
    plt.figure(figsize=(6, 6))

    verified_counts.plot(kind="pie", autopct="%1.1f%%")

    plt.title(f"Proportion des achats vérifiés — {output_name}")
    plt.ylabel("")

    plt.savefig(TASK_ROOT / f"{output_name}_verified_purchase.png", dpi=300, bbox_inches="tight")


# Exécution
task_analyse_distribution(file_path_50k)
task_analyse_distribution(file_path_temp)