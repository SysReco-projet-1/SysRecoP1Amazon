import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz

from pathlib import Path

# GESTION DE L'INPUT/OUTPUT
ROOT = Path(__file__).resolve().parents[1]

OUTPUT = ROOT / "outputs"

SPLITS = OUTPUT / "splits"
FIGURES = OUTPUT / "figures"
MAPPINGS = OUTPUT / "mappings"
MATRIX = OUTPUT / "matrice"

def create_matrix(df, file):
    
    # On fait une copie pour éviter d'altérer les données d'origine
    df = df.copy()

    # Définition des lignes/colonnes
    df["user_idx"] = df["user_id"].astype("category").cat.codes
    df["item_idx"] = df["parent_asin"].astype("category").cat.codes

    # Compte le nombre d'user et d'items distincts de la matrice et les affiche
    n_users = df["user_idx"].nunique()
    n_items = df["item_idx"].nunique()
    print("n_users =", n_users, " | n_items =", n_items, " | ratings =", len(df))

    # Décrit quel uitilisateur correspond a quel ID dans la matrice, conservé sous forme de fichiers
    user_mapping = df[["user_id", "user_idx"]].drop_duplicates().set_index("user_idx")["user_id"]
    item_mapping = df[["parent_asin", "item_idx"]].drop_duplicates().set_index("item_idx")["parent_asin"]

    # Gestion doublons user-item (ex: un user a noté 2 fois un item), on choisit de prendre la moyenne des 2 notes
    df_ui = (
        df.groupby(["user_idx", "item_idx"], as_index=False)["rating"].mean()
    )

    # Construction de la matrice
    rows = df_ui["user_idx"].to_numpy()
    cols = df_ui["item_idx"].to_numpy()
    data = df_ui["rating"].to_numpy(dtype=np.float32)

    R = csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)

    print("R shape:", R.shape)
    print("nnz (non-zeros):", R.nnz)

    # Quelques vérifications de sanité
    # densité (liée à la sparsité)
    density = R.nnz / (R.shape[0] * R.shape[1])
    sparsity = 1 - density
    print("density:", density)
    print("sparsity:", sparsity)

    # Sauvegarde matrice au format NPZ
    save_npz(MATRIX / f"mat_csr_{file}.npz", R)
    
    # et mappings si besoin
    user_mapping.to_csv(MAPPINGS / f"user_mapping_{file}.csv")
    item_mapping.to_csv(MAPPINGS / f"item_mapping_{file}.csv")

    # Visualisation matrice
    plt.figure(figsize=(8, 8))
    plt.spy(R, markersize=0.5)
    plt.title("Structure de la matrice utilisateur–item")
    plt.xlabel("Items")
    plt.ylabel("Utilisateurs")
    plt.savefig(FIGURES / f"matrice_{file}.png", dpi=300, bbox_inches="tight")
    plt.close()



