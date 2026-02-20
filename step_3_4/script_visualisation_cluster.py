import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from matplotlib.colors import BoundaryNorm

from pathlib import Path

# GESTION DE L'INPUT/OUTPUT
ROOT = Path(__file__).resolve().parents[1]

OUTPUT = ROOT / "outputs"

SPLITS = OUTPUT / "splits"
FIGURES = OUTPUT / "figures"
MAPPINGS = OUTPUT / "mappings"
MATRIX = OUTPUT / "matrices"
REPORTS = OUTPUT / "reports"
# ====================================================================
# 3. Visualisation 2D :
# — Utilisez PCA pour réduire à 2 dimensions
# — Créez un scatter plot avec couleurs par cluster
# — Ajoutez les centres des clusters
# ====================================================================

# Fonction effectuant la tâche 3
def task_visualisation(matrix, clusters, kmeans, file_path):

    # Nom pour différencier les fichiers générés
    output_name = file_path.stem

    # Réduction dimensionnelle 2D
    svd = TruncatedSVD(n_components=2, random_state=42)
    matrix_2d = svd.fit_transform(matrix)

    # Centres les clusters en 2D
    centers_2d = svd.transform(kmeans.cluster_centers_)

    # Colormap avec k couleurs exactes
    k = len(centers_2d)
    cmap = plt.cm.get_cmap("tab20", k)

    # Plot
    plt.figure(figsize=(8, 6))

    # Points
    points = plt.scatter(
        matrix_2d[:, 0],
        matrix_2d[:, 1],
        c=clusters,
        cmap=cmap,
        s=8,
        linewidths=0
    )

    # Indices des centres (0,1,2,...,k-1)
    center_labels = np.arange(len(centers_2d))

    # Centres
    centers = plt.scatter(
        centers_2d[:, 0],
        centers_2d[:, 1],
        c=center_labels,
        cmap=cmap,
        marker="X",
        s=180,
        edgecolors="black",
        linewidths=1.5,
        alpha=0.7,
        zorder=5
    )

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(f"Visualisation 2D des clusters — {output_name}")

    # Met des couleurs
    for i in range(k):
        plt.scatter([], [], color=cmap(i), label=f"Cluster {i + 1}")

    plt.legend()

    # Zoom pour mieux voir les clusters (les outliers sont pas affiché du coup mais bon sinon on voit rien sur le graphique)
    plt.xlim(-5, 80)
    plt.ylim(-5, 80)

    plt.tight_layout()

    # Sauvegarde
    output_plot = FIGURES / f"{output_name}_clusters_2d.png"
    plt.savefig(output_plot)
    plt.close()

    print(f"Graphique sauvegardé : {output_plot}")