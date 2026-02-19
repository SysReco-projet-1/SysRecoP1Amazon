from sklearn.model_selection import train_test_split
import pandas as pd

from pathlib import Path

# GESTION DE L'INPUT/OUTPUT
ROOT = Path(__file__).resolve().parents[1]

OUTPUT = ROOT / "outputs"

SPLITS = OUTPUT / "splits"
FIGURES = OUTPUT / "figures"
MAPPINGS = OUTPUT / "mappings"

def create_crossvalid_data(df, file):
    train_parts = []
    test_parts = []
    
    # On boucle par utilisateurs 
    # (Ce qui signifie que l'on split chaque user séparément pour assurer le respect de la consigne)
    # User non utilisé, on le garde pour la lisibilité et potentiel debug
    for user, group in df.groupby("user_id"):

        # Si un utilisateur n'a qu'une intéraction on ne peux pas respecter la consigne, 
        # on met donc arbitrairement dans train.
        if len(group) == 1:
            train_parts.append(group)  # cas extrême
            continue

        # On split le jdd en 80% - 20% (jdd ici étant les évals d'un seul user)
        train_u, test_u = train_test_split(
            group,
            test_size=0.2, # 20% pour le jdd de test
            random_state=42 # Aléatoire mais reproductible (seed=42)
        )

        # On stocke chaque split par user
        train_parts.append(train_u)
        test_parts.append(test_u)

    # On recompose le tout pour ne faire qu'un jdd de chaque type
    train_df = pd.concat(train_parts)
    test_df = pd.concat(test_parts)

    # On save au format csv
    train_df.to_csv(SPLITS / f"train_{file}.csv", index=False)
    test_df.to_csv(SPLITS / f"test_{file}.csv", index=False)

