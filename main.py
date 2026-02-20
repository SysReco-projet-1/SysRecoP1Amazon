import pandas as pd
import sys
from pathlib import Path

# ===================================================
# GESTION DES CHEMINS
# ===================================================

ROOT = Path(__file__).resolve().parent

# TODO : importer vos chemins
sys.path.append(str(ROOT / "step_3_1_3_pretraitement"))
sys.path.append(str(ROOT / "step_3_2_similarity"))


from script_similarity import create_similarity
from script_sparsity import sparsity_rate
from script_matrix import create_matrix
from script_crossvalidation import create_crossvalid_data
from script_pretraitement import (
    pretraitement_ratings,
    pretraitement_timestamps,
    pretraitement_filtrage_iteratif,
)

# ===================================================
# GESTION DE L'INPUT/OUTPUT
# ===================================================

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
# FICHIERS
# ===================================================

file1 = "amazon_books_sample_active_users.csv"
file2 = "amazon_books_sample_temporal.csv"

# ===================================================
# CHARGEMENT
# ===================================================

def lecture_fichier(file):
    print("=" * 10 + "\nLECTURE...\nFichier: ", file, "\n" + "=" * 10)
    df = pd.read_csv(INPUT / file)
    return df


# ===================================================
# PRÉTRAITEMENT COMPLET
# ===================================================

def pretraitement_complet(df1, df2):
    """Applique tout le pipeline de prétraitement sur les deux fichiers."""

    print("=" * 5 + "\nSparsité pré nettoyage\n" + file1 + "\n" + "=" * 5)
    print(sparsity_rate(df1))
    print("=" * 5 + "\nSparsité pré nettoyage\n" + file2 + "\n" + "=" * 5)
    print(sparsity_rate(df2))

    print("=" * 5 + " GESTION RATINGS " + "=" * 5)
    df1 = pretraitement_ratings(df1, file1)
    print("=" * 5 + " CHANGEMENT DE FICHIER " + "=" * 5)
    df2 = pretraitement_ratings(df2, file2)

    print("=" * 5 + " GESTION TIMESTAMPS " + "=" * 5)
    df1 = pretraitement_timestamps(df1, file1)
    print("=" * 5 + " CHANGEMENT DE FICHIER " + "=" * 5)
    df2 = pretraitement_timestamps(df2, file2)

    print("=" * 5 + " FILTRAGE " + "=" * 5)
    df1 = pretraitement_filtrage_iteratif(df1)
    print("=" * 5 + " CHANGEMENT DE FICHIER " + "=" * 5)
    df2 = pretraitement_filtrage_iteratif(df2)

    print("=" * 5 + "\nSparsité post nettoyage\n" + file1 + "\n" + "=" * 5)
    print(sparsity_rate(df1))
    print("=" * 5 + "\nSparsité post nettoyage\n" + file2 + "\n" + "=" * 5)
    print(sparsity_rate(df2))

    return df1, df2


# ===================================================
# TÂCHES INDIVIDUELLES
# ===================================================

def run_tache_0(df1, df2):
    """Tâche 0 - Chargement, échantillonnage, prétraitement, matrice, split."""
    df1, df2 = pretraitement_complet(df1, df2)

    print("=" * 5 + "\nCréation de la matrice CSR\n" + file1 + "\n" + "=" * 5)
    create_matrix(df1, file1)
    print("=" * 5 + "\nCréation de la matrice CSR\n" + file2 + "\n" + "=" * 5)
    create_matrix(df2, file2)

    print("=" * 5 + "\nCréation des données de validation croisée\n" + file1 + "\n" + "=" * 5)
    create_crossvalid_data(df1, file1)
    print("=" * 5 + "\nCréation des données de validation croisée\n" + file2 + "\n" + "=" * 5)
    create_crossvalid_data(df2, file2)

    return df1, df2


def run_tache_1(df1, df2):
    """Tâche 1 - Mesures de similarité."""
    create_similarity(df1, file1)
    create_similarity(df2, file2)


def run_tache_2(df1, df2):
    """Tâche 2 - Représentation en graphe biparti."""
    create_graph(df1, file1)
    create_graph(df2, file2)


def run_tache_3(df1, df2):
    """Tâche 3 - Regroupement des utilisateurs."""
    # TODO : importer et appeler script_clustering.py
    print("  (Tâche 3 non encore implémentée)")


def run_tache_4(df1, df2):
    """Tâche 4 - Prédiction des évaluations."""
    # TODO : importer et appeler script_prediction.py
    print("  (Tâche 4 non encore implémentée)")


def run_tache_5(df1, df2):
    """Tâche 5 - Discussion et analyse critique."""
    # TODO : importer et appeler script_discussion.py
    print("  (Tâche 5 non encore implémentée)")


# ===================================================
# MENU INTERACTIF
# ===================================================

# Rajout d'option dans le menu
ETAPES = {
    "0": ("Tâche 0 - Chargement, échantillonnage et prétraitement", run_tache_0),
    "1": ("Tâche 1 - Mesures de similarité",                        run_tache_1),
    "2": ("Tâche 2 - Représentation en graphe biparti",             run_tache_2),
    "3": ("Tâche 3 - Regroupement des utilisateurs (K-Means)",      run_tache_3),
    "4": ("Tâche 4 - Prédiction des évaluations (k-NN)",            run_tache_4),
}


def afficher_menu():
    print("\n" + "=" * 55)
    print("   INF6083 — Systèmes de recommandation — Projet P1")
    print("=" * 55)
    for key, (label, _) in ETAPES.items():
        print(f"  [{key}] {label}")
    print("  [a] Exécuter toutes les tâches")
    print("  [q] Quitter")
    print("=" * 55)


def menu(df1, df2):
    while True:
        afficher_menu()
        choix = input("\nQuelle étape voulez-vous exécuter ? ").strip().lower()

        if choix == "q":
            print("\nAu revoir !\n")
            break

        elif choix == "a":
            print("\n→ Exécution de toutes les tâches...\n")
            for key, (label, fonction) in ETAPES.items():
                print(f"\n{'=' * 55}\n→ {label}\n{'=' * 55}")
                result = fonction(df1, df2)
                if key == "0" and result is not None:
                    df1, df2 = result

        elif choix in ETAPES:
            label, fonction = ETAPES[choix]
            print(f"\n→ Exécution : {label}\n")
            result = fonction(df1, df2)
            if choix == "0" and result is not None:
                df1, df2 = result

        else:
            print("\n  Choix invalide, veuillez réessayer.")

    return df1, df2


# ===================================================
# POINT D'ENTRÉE
# ===================================================

def main():
    df1  = lecture_fichier(file1)
    df2 = lecture_fichier(file2)
    menu(df1, df2)


if __name__ == "__main__":
    main()