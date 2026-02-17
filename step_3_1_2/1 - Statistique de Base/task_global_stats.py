import pandas as pd
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

# =========================================================================================
# 1. Statistiques de base
# - Nombre total d’utilisateurs, de livres et d’évaluations
# - Distribution des évaluations (histogramme)
# - Nombre moyen d’évaluations par utilisateur et par livre
# - Identification des 10 utilisateurs les plus actifs et des 10 livres les plus populaires
# =========================================================================================

# ==============================
# Fonction effectuant la tâche 1
# ==============================
def task_global_stats(file_path):

    # Nom pour différentier les fichier csv source
    output_name = file_path.stem

    # Charger le fichier CSV
    df = pd.read_csv(file_path)

    # Nombre total de ratings
    num_ratings = len(df)

    # Nombre d'utilisateurs uniques
    num_users = df["user_id"].nunique()

    # Nombre d'items (livres) uniques
    num_items = df["parent_asin"].nunique()

    # Distribution des évaluations
    rating_distribution = df["rating"].value_counts().sort_index()

    # Moyennes par utilisateur et par livre
    ratings_per_user = df.groupby("user_id").size()
    ratings_per_item = df.groupby("parent_asin").size()

    avg_ratings_per_user = ratings_per_user.mean()
    avg_ratings_per_item = ratings_per_item.mean()

    # Top utilisateurs et livres
    top_users = ratings_per_user.sort_values(ascending=False).head(10)
    top_items = ratings_per_item.sort_values(ascending=False).head(10)

    # ===========================
    # Écriture dans fichier texte
    # ===========================
    output_txt = TASK_ROOT / f"{output_name}_data.txt"

    with open(output_txt, "w", encoding="utf-8") as f:

        def write(line=""):
            print(line)          # console
            f.write(line + "\n") # fichier

        write("===== Nombre total d’utilisateurs, de livres et d’évaluations =====")
        write(f"Nombres total d'utilisateurs : {num_users}")
        write(f"Nombres total de livres      : {num_items}")
        write(f"Nombres total d'évaluations  : {num_ratings}")
        write()

        write("===== Distribution des évaluations =====")
        write(rating_distribution.to_string())
        write()

        write("===== Nombre moyen d’évaluations =====")
        write(f"Par utilisateur : {avg_ratings_per_user:.2f}")
        write(f"Par livre       : {avg_ratings_per_item:.2f}")
        write()

        write("===== Top 10 Utilisateurs =====")
        write(top_users.to_string())
        write()

        write("===== Top 10 Livres =====")
        write(top_items.to_string())

    # ==============================================
    # Histogramme de la distribution des évaluations
    # ==============================================

    # Taille du graphique
    plt.figure(figsize=(8, 5))
    plt.hist(df["rating"], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])

    # Titre et labels
    plt.title(f"Distribution des évaluations ({output_name})")
    plt.xlabel("Rating")
    plt.ylabel("Nombre d'occurrences")

    plt.xticks([1, 2, 3, 4, 5])

    # Sauvegarde du graphique du format png
    plt.savefig(TASK_ROOT / f"rating_distribution_{output_name}.png", dpi=300, bbox_inches="tight")

# Éxécution
task_global_stats(file_path_50k)
task_global_stats(file_path_temp)