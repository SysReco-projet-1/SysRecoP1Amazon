import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# GESTION DE L'INPUT/OUTPUT
ROOT = Path(__file__).resolve().parents[1]

OUTPUT = ROOT / "outputs"

SPLITS = OUTPUT / "splits"
FIGURES = OUTPUT / "figures"
MAPPINGS = OUTPUT / "mappings"
MATRIX = OUTPUT / "matrices"
REPORTS = OUTPUT / "reports"

# =============================
# 2. Calculer taux de  sparsité
# =============================

# Fonction effectuant la tâche 2
def task_sparsity_global(df, file_path):

    # Nom pour différentier les fichier csv source
    output_name = Path(file_path).stem

    # Récupère les colonnes nécéssaires
    n_users = df["user_id"].nunique()
    n_items = df["parent_asin"].nunique()
    n_ratings = len(df)

    # Calcule la sparsité
    sparsity = 1 - (n_ratings / (n_users * n_items))

    # ===========================
    # Écriture dans fichier texte
    # ===========================
    output_txt = REPORTS / f"{output_name}_sparsity_global_stat.txt"

    with open(output_txt, "w", encoding="utf-8") as f:

        def write(line=""):
            print(line)          # console
            f.write(line + "\n") # fichier

        write("===== Taux de sparsité =====")
        write(f"Nombres d'utilisateurs : {n_users}")
        write(f"Nombres de livres : {n_items}")
        write(f"Nombres d'évaluations : {n_ratings}")
        write(f"Taux de sparsité : {sparsity}")