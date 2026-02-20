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
MATRIX = OUTPUT / "matrices"

SPLITS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
MAPPINGS.mkdir(parents=True, exist_ok=True)
MATRIX.mkdir(parents=True, exist_ok=True)


# ===================================================
    # Prétraitement des données
# ===================================================

def pretraitement_ratings(df, file):
    # ===================================================
        # Etape 1 : Suppression des reviews sans ratings
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