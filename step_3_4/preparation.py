import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz
from pathlib import Path
from sklearn.preprocessing import StandardScaler

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

# ===============================================================
# 1. Préparation des données :
# — Sélectionnez un échantillon de 10,000 utilisateurs aléatoires
# — Normalisez les vecteurs utilisateur avec StandardScaler
# — Justifiez le choix de la taille d’échantillon
# ===============================================================

# Fonction effectuant la tâche 1
def task_preparation(file_path):

    # Nom pour différentier les fichier csv source
    output_name = file_path.stem

    # Charger le fichier CSV
    df = pd.read_csv(file_path)
    print(f"\nChargement csv : {file_path}")

    # Tous les Utilisateurs uniques
    users = df["user_id"].unique()

    # On sélectionne 10 000 utilisateurs aléatoire avec la seed 42 (réponse à la vie)
    sample_size = min(10_000, len(users))
    sampled_users = pd.Series(users).sample(sample_size, random_state=42)

    df_sample = df[df["user_id"].isin(sampled_users)]

    # ==============================
    # Construction matrice user-item
    # ==============================

    df_sample = df_sample.copy()

    # Encodage indices
    df_sample["user_idx"] = df_sample["user_id"].astype("category").cat.codes
    df_sample["item_idx"] = df_sample["parent_asin"].astype("category").cat.codes

    # Sauvegarde le mapping catégorie → id pour le clustering plus tard
    user_categories = df_sample["user_id"].astype("category").cat.categories
    item_categories = df_sample["parent_asin"].astype("category").cat.categories

    np.save(OUTPUT_ROOT / f"{output_name}_user_ids.npy", user_categories.to_numpy())
    np.save(OUTPUT_ROOT / f"{output_name}_item_ids.npy", item_categories.to_numpy())

    n_users = df_sample["user_idx"].nunique()
    n_items = df_sample["item_idx"].nunique()

    # Gestion doublons
    df_ui = df_sample.groupby(["user_idx", "item_idx"], as_index=False)["rating"].mean()

    rows = df_ui["user_idx"].to_numpy()
    cols = df_ui["item_idx"].to_numpy()
    data = df_ui["rating"].to_numpy(dtype=np.float32)

    R = csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)

    # Sauvegarde de la matrice
    npz_path = OUTPUT_ROOT / f"{output_name}_user_item_matrix"
    save_npz(npz_path, R)
    print(f"Sauvegarde npz : {output_name}_user_item_matrix.npz")

    # ==========================
    # Normalisation des vecteurs
    # ==========================
    scaler = StandardScaler(with_mean=False)
    R_norm = scaler.fit_transform(R)

    save_npz(f"{npz_path}_normalized", R_norm)
    print(f"Sauvegarde npz : {output_name}_user_item_matrix_normalized.npz")

    # ===========================================
    # Justification de la taille des échantillons
    # ===========================================
    output_txt = OUTPUT_ROOT / f"{output_name}_justification_sample.txt"

    with open(output_txt, "w", encoding="utf-8") as f:

        def write(line=""):
            print(line)          # console
            f.write(line + "\n") # fichier

        write("===== Préparation des données =====")
        write(f"Nombre total utilisateurs : {len(users)}")
        write(f"Utilisateurs échantillonnés : {sample_size}")
        write(f"Nombre de livres : {R.shape[1]}")
        write(f"Taille matrice : {R.shape}")
        write()
        write("La normalisation a été réalisée à l’aide de StandardScaler sans centrage afin de préserver la structure creuse de la matrice sparse, le centrage introduisant des valeurs non nulles incompatibles avec ce format.")
        write()
        write("Justification taille échantillon :")
        write("Un échantillon de 10 000 utilisateurs permet de réduire le coût computationnel et est suffisante pour capturer l’hétérogénéité des comportements utilisateurs.")
        write("Elle constitue ainsi une approximation réaliste du jeu de données complet tout en restant exploitable dans un environnement expérimental.")