from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD

from pathlib import Path

# GESTION DE L'INPUT/OUTPUT
ROOT = Path(__file__).resolve().parents[1]

OUTPUT = ROOT / "outputs"

SPLITS = OUTPUT / "splits"
FIGURES = OUTPUT / "figures"
MAPPINGS = OUTPUT / "mappings"
MATRIX = OUTPUT / "matrices"
REPORTS = OUTPUT / "reports"
 
# =======================================================================
# 1. Profils des clusters : Pour chaque cluster identifié, calculez :
# — Taille du cluster
# — Centre du cluster
# — Évaluation moyenne des utilisateurs du cluster
# — Écart-type des évaluations
# — Les 10 livres préférés du cluster (moyennes les plus élevées)
# =======================================================================

# Fonction effectuant la tâche 1
def task_cluster_profile(df, matrix, kmeans, user_ids, file_path):

    # Nom pour différencier les fichiers
    output_name = file_path.stem

    # Mapping item index -> parent_asin
    item_ids = df["parent_asin"].unique()

    # Réduction pour éviter d'avoir un seul gros cluster
    svd = TruncatedSVD(n_components=100, random_state=42)
    matrix_reduced = svd.fit_transform(matrix)

    # Cluster
    clusters = kmeans.fit_predict(matrix_reduced)
    n_clusters = len(np.unique(clusters))

    results = []

    # Sauvegarde texte
    output_file = REPORTS / f"{output_name}_clusters.txt"

    with open(output_file, "w", encoding="utf-8") as f:

        def write(line=""):
            print(line)          # console
            f.write(line + "\n") # fichier

        # =========================
        # Analyse par cluster
        # =========================
        for i in range(n_clusters):

            print(f"Analyse cluster {i + 1}")

            cluster_indices = np.where(clusters == i)[0]

            # Taille du cluster
            cluster_size = len(cluster_indices)

            if cluster_size == 0:
                continue

            # Sous matrice cluster
            cluster_matrix = matrix[cluster_indices]

            # Centre du cluster μk
            centroid = cluster_matrix.mean(axis=0)
            centroid = np.asarray(centroid).flatten()

            # Stats ratings
            cluster_users = user_ids[cluster_indices]

            df_cluster = df[df["user_id"].isin(cluster_users)]

            mean_rating = df_cluster["rating"].mean()
            std_rating = df_cluster["rating"].std()

            # Top livres du cluster
            top_books = (
                df_cluster
                .groupby("parent_asin")["rating"]
                .count()
                .sort_values(ascending=False)
                .head(10)
            )

            write(f"Cluster {i + 1}")
            write('=' * 60)

            write(f"Taille du cluster : {cluster_size}")
            write(f"Centre du cluster : {centroid}")
            write(f"Moyenne ratings : {mean_rating:.4f}")
            write(f"Écart-type ratings : {std_rating:.4f}")

            write("Top 10 livres :")
            for book, count in top_books.items():
                write(f"{book} : {count}")
            write()
            write('=' * 60)

    print(f"Résultats sauvegardés : {output_file}")

    return  matrix_reduced, clusters