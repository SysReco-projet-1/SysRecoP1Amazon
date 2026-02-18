import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# =====================================
# Variables de l'arboréscence du projet
# =====================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = Path(__file__).resolve().parent
FILE_NAME = Path(__file__).resolve().stem

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / TASK_ROOT.name / FILE_NAME
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Chemin vers les fichiers csv prétraité : Splits
SPLITS = Path("outputs") / "splits"

file_path_50k_train = PROJECT_ROOT / SPLITS / "train_amazon_books_sample_active_users.csv"
file_path_50k_test = PROJECT_ROOT / SPLITS / "test_amazon_books_sample_active_users.csv"

file_path_temp_train = PROJECT_ROOT / SPLITS / "train_amazon_books_sample_temporal.csv"
file_path_temp_test = PROJECT_ROOT / SPLITS / "test_amazon_books_sample_temporal.csv"

# ============================================================================
# 1. Préparation des données :
# — Sélectionnez un échantillon de 10,000 utilisateurs aléatoires
# — Normalisez les vecteurs utilisateur avec StandardScaler
# — Justifiez le choix de la taille d’échantillon
# ============================================================================

# Fonction effectuant la tâche 1
def task_preparation(file_path):

    # Nom pour différentier les fichier csv source
    output_name = file_path.stem

    # Charger le fichier CSV
    df = pd.read_csv(file_path)

    # Tous les Utilisateurs uniques
    users = df["user_id"].unique()

    # On sélectionne 10 000 utilisateurs aléatoire avec la seed 42 (réponse à la vie)
    sample_size = min(10_000, len(users))
    sampled_users = pd.Series(users).sample(sample_size, random_state=42)

    df_sample = df[df["user_id"].isin(sampled_users)]

    print(len(sampled_users))
    print(len(df_sample))

# Exécution
task_preparation(file_path_50k_train)
task_preparation(file_path_temp_train)