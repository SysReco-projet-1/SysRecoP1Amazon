import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# =====================================
# Variables de l'arboréscence du projet
# =====================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = Path(__file__).resolve().parent
FILE_NAME = Path(__file__).resolve().stem

OUTPUT_ROOT = PROJECT_ROOT / "output" / TASK_ROOT.name / FILE_NAME
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Chemin vers les deux fichiers csv
file_path_50k = PROJECT_ROOT / "input" / "amazon_books_sample_active_users.csv"
file_path_temp = PROJECT_ROOT / "input" / "amazon_books_sample_temporal.csv"

# =============================
# 2. Calculer taux de  sparsité
# =============================

# Fonction effectuant la tâche 2
def task_sparsity(file_path):

    # Nom pour différentier les fichier csv source
    output_name = file_path.stem

    # Charger le fichier CSV
    df = pd.read_csv(file_path)

    # Récupère les colonnes nécéssaires
    n_users = df["user_id"].nunique()
    n_items = df["parent_asin"].nunique()
    n_ratings = len(df)

    # Calcule la sparsité
    sparsity = 1 - (n_ratings / (n_users * n_items))

    # ===========================
    # Écriture dans fichier texte
    # ===========================
    output_txt = OUTPUT_ROOT / f"{output_name}_{FILE_NAME}.txt"

    with open(output_txt, "w", encoding="utf-8") as f:

        def write(line=""):
            print(line)          # console
            f.write(line + "\n") # fichier

        write("===== Taux de sparsité =====")
        write(f"Nombres d'utilisateurs : {n_users}")
        write(f"Nombres de livres : {n_items}")
        write(f"Nombres d'évaluations : {n_ratings}")
        write(f"Taux de sparsité : {sparsity}")

# Éxécution
task_sparsity(file_path_50k)
task_sparsity(file_path_temp)