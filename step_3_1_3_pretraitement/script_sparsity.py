import pandas as pd

def sparsity_rate(df):
    # Récupère les colonnes nécéssaires
    n_users = df["user_id"].nunique()
    n_items = df["parent_asin"].nunique()
    n_ratings = len(df)

    # Calcule la sparsité
    sparsity = 1 - (n_ratings / (n_users * n_items))

    return n_users, n_items, n_ratings, sparsity