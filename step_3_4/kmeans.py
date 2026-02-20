import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# =====================================
# Variables de l'arboréscence du projet
# =====================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = Path(__file__).resolve().parent
FILE_NAME = Path(__file__).resolve().stem

# Où on met les fichiers
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / TASK_ROOT.name / FILE_NAME
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Chemin vers les fichiers csv prétraité : Splits
SPLITS = Path("outputs") / "splits"

file_path_50k_train = PROJECT_ROOT / SPLITS / "train_amazon_books_sample_active_users.csv"
file_path_50k_test = PROJECT_ROOT / SPLITS / "test_amazon_books_sample_active_users.csv"

file_path_temp_train = PROJECT_ROOT / SPLITS / "train_amazon_books_sample_temporal.csv"
file_path_temp_test = PROJECT_ROOT / SPLITS / "test_amazon_books_sample_temporal.csv"

file_path_50k_matrix = PROJECT_ROOT / "outputs" / TASK_ROOT.name / "preparation" / "train_amazon_books_sample_active_users_user_item_matrix_normalized.npz"
file_path_temp_matrix = PROJECT_ROOT / "outputs" / TASK_ROOT.name / "preparation" / "train_amazon_books_sample_temporal_user_item_matrix_normalized.npz"

# ========================================================================================================================================
# 2. K-Means pour différents K : Appliquez K-Means pour K ∈ {3, 4, 5, 6, 7, 8}.
# Pour chaque K, calculez le score de Silhouette (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
# et affichez le diagramme en barre
# ========================================================================================================================================

# Fonction effectuant la tâche 2
def task_kmeans(file_path, K_range=range(3, 9)):

    # Nom pour différencier les fichiers générés
    output_name = file_path.stem

    print(f"\nChargement matrice : {file_path}")

    # On charge la matrice
    X = load_npz(file_path)

    output_txt = OUTPUT_ROOT / f"{output_name}_silhouette_scores.txt"

    silhouette_scores = []

    best_kmeans = None
    best_score = -1
    best_k = None

    with open(output_txt, "w", encoding="utf-8") as f:

        def write(line=""):
            print(line)
            f.write(line + "\n")

        for k in K_range:

            write(f"KMeans avec K={k}")

            kmeans = MiniBatchKMeans(
                n_clusters=k,
                random_state=42,
            )

            labels = kmeans.fit_predict(X)

            score = silhouette_score(X, labels)

            silhouette_scores.append((k, score))

            write(f"Silhouette score K={k} : {score:.4f}")

            # Mise à jour du meilleur modèle
            if score > best_score:
                best_score = score
                best_k = k
                best_kmeans = kmeans

        write()
        write(f"Meilleur K : {best_k}")
        write(f"Silhouette max : {best_score:.4f}")

    # Graphique
    plt.figure(figsize=(8, 5))

    ks, scores = zip(*silhouette_scores)

    plt.bar(ks, scores)

    plt.xlabel("Nombre de clusters (K)")
    plt.ylabel("Score de Silhouette")
    plt.title(f"KMeans Silhouette Scores — {output_name}")

    plt.tight_layout()

    output_plot = OUTPUT_ROOT / f"{output_name}_silhouette_scores.png"
    plt.savefig(output_plot)
    plt.close()

    print(f"Graphique sauvegardé : {output_plot}")

    # Retour du meilleur K-Means
    return best_kmeans