from preparation import task_preparation
from kmeans import task_kmeans
from cluster_profile import task_cluster_profile
from visualisation import  task_visualisation
from pathlib import Path

# =====================================
# Variables de l'arboréscence du projet
# =====================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = Path(__file__).resolve().parent
FILE_NAME = Path(__file__).resolve().stem

# Chemin vers les fichiers csv prétraité : Splits
SPLITS = Path("outputs") / "splits"

file_path_50k_train = PROJECT_ROOT / SPLITS / "train_amazon_books_sample_active_users.csv"
file_path_50k_test = PROJECT_ROOT / SPLITS / "test_amazon_books_sample_active_users.csv"

file_path_temp_train = PROJECT_ROOT / SPLITS / "train_amazon_books_sample_temporal.csv"
file_path_temp_test = PROJECT_ROOT / SPLITS / "test_amazon_books_sample_temporal.csv"

file_path_50k_matrix = PROJECT_ROOT / "outputs" / TASK_ROOT.name  / "preparation" / "train_amazon_books_sample_active_users_user_item_matrix_normalized.npz"
file_path_temp_matrix = PROJECT_ROOT / "outputs" / TASK_ROOT.name / "preparation" / "train_amazon_books_sample_temporal_user_item_matrix_normalized.npz"

file_path_user_ids = PROJECT_ROOT / "outputs" / TASK_ROOT.name / "preparation"

def task():
    
    # On prépare les données
    print(f"Préparation des données..."+ "\n")
    task_preparation(file_path_50k_train)
    task_preparation(file_path_temp_train)
    print(f"Préparation des données..."+ "\n")

    # On fait les K-Means et on récupère le meilleur que l'on va utiliser pour le cluster
    print(f"Calcul des K-Means..."+ "\n")
    kmean_50k = task_kmeans(file_path_50k_matrix)
    kmean_temp = task_kmeans(file_path_temp_matrix)
    print(f"Calcul des K-Means : done !"+ "\n")

    # On fait le cluster
    print(f"Création des clusters du meilleur K-Means..."+ "\n")
    matrix_50k, cluster_50k = task_cluster_profile(file_path_50k_matrix, file_path_50k_train, kmean_50k)
    matrix_temp, cluster_temp = task_cluster_profile(file_path_temp_matrix, file_path_temp_train, kmean_temp)
    print(f"Création des clusters du meilleur K-Means : done !"+ "\n")

    # On fait la visualisation 2D
    print(f"Visualisation 2D..."+ "\n")
    task_visualisation(matrix_50k, cluster_50k, kmean_50k, file_path_50k_matrix)
    task_visualisation(matrix_temp, cluster_temp, kmean_temp, file_path_temp_matrix)
    print(f"Visualisation 2D : done !"+ "\n")

task()