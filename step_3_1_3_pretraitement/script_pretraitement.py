import pandas as pd
from script_sparsity import sparsity_rate
from script_matrix import create_matrix
from script_crossvalidation import create_crossvalid_data

from pathlib import Path

# GESTION DE L'INPUT/OUTPUT
ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "input"
OUTPUT = ROOT / "outputs"

SPLITS = OUTPUT / "splits"
FIGURES = OUTPUT / "figures"
MAPPINGS = OUTPUT / "mappings"
MATRIX = OUTPUT / "matrice"

SPLITS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
MAPPINGS.mkdir(parents=True, exist_ok=True)
MATRIX.mkdir(parents=True, exist_ok=True)


# ===================================================
    # Prétraitement des données
# ===================================================


file1 = "amazon_books_sample_active_users"
file2 = "amazon_books_sample_temporal"


def lecture_fichier(file):
    # Lis les fihciers CSV contenant les données échantillonnées
    print("==============\nLECTURE...\nFichier: ", file, "\n==============")
    df = pd.read_csv(INPUT / f"{file}.csv")
    return df


def pretraitement_ratings(df, file):
    # ===================================================
        # Etape 1 : Suppression des reveiws sans ratings
    # ===================================================

    # Toujours visualiser ce que l'on néttoie
    print("=======================\nAvant nettoyage\nFichier : ", file, "\n=======================")
    print("Nombre total de reviews :", len(df))
    print("NaN : ", df["rating"].isna().sum())

    # Conversion des ratings en float
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Suppressions des ratings manquants (r=0) et hors plage (1 < r < 5)
    df = df.dropna(subset=["rating"])
    df = df[(df["rating"] >= 1) & (df["rating"] <= 5)]

    # Visualisation après nettoyage
    print("\n")
    print("=======================\nAprès nettoyage\nFichier : ", file, "\n=======================")
    print("Reviews restants :", len(df))
    print("Min rating :", df["rating"].min())
    print("Max rating :", df["rating"].max())
    print("NaN restants :", df["rating"].isna().sum())

    return df


def pretraitement_timestamps(df, file):
    # ===================================================
        # Etape 2 : Suppression des timestamps invalides
    # ===================================================

    # Conversion des timestamps en datetime (inavlides -> NaT)
    df["date"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")

    # Visualisation des timestamps avant nettoyage
    print("=======================\nAvant nettoyage\nFichier : ", file, "\n=======================")
    print("Timestamps vides :", df["date"].isna().sum())
    print("Date min :", df["date"].min())
    print("Date max :", df["date"].max())

    # Suppression des timestamps invalides (NaT)
    df = df.dropna(subset=["date"])
    
    # Filtrage des dates absurdes (avant mai 1996 ou après aujourd'hui)
    df = df[(df["date"] >= "1996-05-01") & (df["date"] <= pd.Timestamp.today())]
    
    # Visualisation après nettoyage
    print("=======================\nAprès nettoyage\nFichier : ", file, "\n=======================")
    print("Date min :", df["date"].min())
    print("Date max :", df["date"].max())

    return df


def pretraitement_filtrage_iteratif(df, min_user=10, min_item=5):
    while True:
        before = len(df)
        print("Before", before)

        # Filtrage des utilisateurs
        user_count = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_count[user_count >= min_user].index)]

        # Filtrage des items
        item_counts = df["parent_asin"].value_counts()
        df = df[df["parent_asin"].isin(item_counts[item_counts >= min_item].index)]

        after = len(df)
        print("After", after)

        if before == after:
            break

    return df


def main():
    # Lecture des fichiers CSV
    df = lecture_fichier(file1)
    df2 = lecture_fichier(file2)

    print("\n==========\nSparsité pre nettoyage\n", file1,"\n==========\n")
    print(sparsity_rate(df))
    print("\n==========\nSparsité pre nettoyage\n", file2,"\n==========\n")
    print(sparsity_rate(df2))

    print("\n========== GESTION RATINGS ==========\n")
    df = pretraitement_ratings(df, file1)
    print("\n========== CHANGEMENT DE FICHIER ==========\n")
    df2 = pretraitement_ratings(df2, file2)

    print("\n========== GESTION TIMESTAMPS ==========\n")
    df = pretraitement_timestamps(df, file1)
    print("\n========== CHANGEMENT DE FICHIER ==========\n")
    df2 = pretraitement_timestamps(df2, file2)

    print("\n========== FILTRAGE ==========\n")
    df = pretraitement_filtrage_iteratif(df)
    print("\n========== CHANGEMENT DE FICHIER ==========\n")
    df2 = pretraitement_filtrage_iteratif(df2)

    print("\n==========\nSparsité post nettoyage\n", file1,"\n==========\n")
    print(sparsity_rate(df))
    print("\n==========\nSparsité post nettoyage\n", file2,"\n==========\n")
    print(sparsity_rate(df2))

    print("\n==========\nCreation de la matrice CSR\n", file1,"\n==========\n")
    create_matrix(df, file1)
    print("\n==========\nCreation de la matrice CSR\n", file2,"\n==========\n")
    create_matrix(df2, file2)

    print("\n==========\nCreation des données de validation croisée\n", file1,"\n==========\n")
    create_crossvalid_data(df, file1)
    print("\n==========\nCreation des données de validation croisée\n", file2,"\n==========\n")
    create_crossvalid_data(df2, file2)


if __name__ == "__main__":
    main()