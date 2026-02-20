import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import json

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
# 1. Modèle Baseline 1 : Moyenne globale
# — Prédire r̂_u,i = r̄ (moyenne globale de toutes les évaluations)
# — Calculer RMSE et MAE sur l'ensemble de test
# ============================================================================

def baseline_global_mean(train_df, test_df):
    """
    Modèle baseline 1 : Moyenne globale
    Prédit la moyenne globale de toutes les évaluations pour chaque paire (user, item)
    
    Args:
        train_df: DataFrame d'entraînement avec colonnes ['user_id', 'parent_asin', 'rating']
        test_df: DataFrame de test avec colonnes ['user_id', 'parent_asin', 'rating']
    
    Returns:
        dict: Dictionnaire contenant RMSE, MAE et temps d'exécution
    """
    start_time = time.time()
    
    # Calcul de la moyenne globale sur l'ensemble d'entraînement
    global_mean = train_df['rating'].mean()
    
    # Prédiction : toutes les prédictions sont égales à la moyenne globale
    predictions = np.full(len(test_df), global_mean)
    
    # Évaluations réelles
    true_ratings = test_df['rating'].values
    
    # Calcul des métriques
    rmse = np.sqrt(mean_squared_error(true_ratings, predictions))
    mae = mean_absolute_error(true_ratings, predictions)
    
    execution_time = time.time() - start_time
    
    results = {
        'model': 'Baseline - Moyenne globale',
        'global_mean': global_mean,
        'rmse': rmse,
        'mae': mae,
        'time_seconds': execution_time,
        'n_predictions': len(test_df)
    }
    
    print(f"Baseline 1 - Moyenne globale:")
    print(f"  Moyenne globale: {global_mean:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Temps d'exécution: {execution_time:.4f}s")
    
    return results


# ============================================================================
# 2. Modèle Baseline 2 : Moyenne par livre
# — Prédire r̂_u,i = r̄_i (moyenne des évaluations du livre i)
# — Si le livre n'est pas dans l'ensemble d'entraînement, utiliser r̄
# — Calculer RMSE et MAE sur l'ensemble de test
# ============================================================================

def baseline_item_mean(train_df, test_df):
    """
    Modèle baseline 2 : Moyenne par livre
    Prédit la moyenne des évaluations pour chaque livre
    Si le livre n'est pas dans le train set, utilise la moyenne globale
    
    Args:
        train_df: DataFrame d'entraînement avec colonnes ['user_id', 'parent_asin', 'rating']
        test_df: DataFrame de test avec colonnes ['user_id', 'parent_asin', 'rating']
    
    Returns:
        dict: Dictionnaire contenant RMSE, MAE et temps d'exécution
    """
    start_time = time.time()
    
    # Calcul de la moyenne globale (fallback pour les livres inconnus)
    global_mean = train_df['rating'].mean()
    
    # Calcul de la moyenne par livre sur l'ensemble d'entraînement
    item_means = train_df.groupby('parent_asin')['rating'].mean().to_dict()
    
    # Prédiction pour chaque évaluation du test set
    predictions = []
    for _, row in test_df.iterrows():
        item_id = row['parent_asin']
        # Si le livre est connu, utiliser sa moyenne, sinon utiliser la moyenne globale
        pred = item_means.get(item_id, global_mean)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    true_ratings = test_df['rating'].values
    
    # Calcul des métriques
    rmse = np.sqrt(mean_squared_error(true_ratings, predictions))
    mae = mean_absolute_error(true_ratings, predictions)
    
    execution_time = time.time() - start_time
    
    # Statistiques sur les livres inconnus
    n_unknown_items = sum(1 for item in test_df['parent_asin'] if item not in item_means)
    unknown_ratio = n_unknown_items / len(test_df)
    
    results = {
        'model': 'Baseline - Moyenne par livre',
        'global_mean': global_mean,
        'n_items_in_train': len(item_means),
        'n_unknown_items_in_test': n_unknown_items,
        'unknown_ratio': unknown_ratio,
        'rmse': rmse,
        'mae': mae,
        'time_seconds': execution_time,
        'n_predictions': len(test_df)
    }
    
    print(f"\nBaseline 2 - Moyenne par livre:")
    print(f"  Nombre de livres dans le train: {len(item_means)}")
    print(f"  Livres inconnus dans le test: {n_unknown_items} ({unknown_ratio*100:.2f}%)")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Temps d'exécution: {execution_time:.4f}s")
    
    return results


# ============================================================================
# 3. Fonction principale pour évaluer les baselines
# — Charge les données train/test
# — Exécute les deux modèles baseline
# — Compare les résultats
# — Sauvegarde les résultats
# ============================================================================

def evaluate_baselines(file_path_train, file_path_test, output_name):
    """
    Évalue les deux modèles baseline sur un jeu de données
    
    Args:
        file_path_train: Chemin vers le fichier CSV d'entraînement
        file_path_test: Chemin vers le fichier CSV de test
        output_name: Nom pour différentier les fichiers de sortie
    """
    print(f"\n{'='*80}")
    print(f"Évaluation des baselines sur: {output_name}")
    print(f"{'='*80}\n")
    
    # Chargement des données
    print("Chargement des données...")
    train_df = pd.read_csv(file_path_train)
    test_df = pd.read_csv(file_path_test)
    
    print(f"Train set: {len(train_df)} évaluations")
    print(f"Test set: {len(test_df)} évaluations")
    print(f"Utilisateurs uniques (train): {train_df['user_id'].nunique()}")
    print(f"Livres uniques (train): {train_df['parent_asin'].nunique()}")
    print()
    
    # Évaluation Baseline 1 : Moyenne globale
    results_global = baseline_global_mean(train_df, test_df)
    
    # Évaluation Baseline 2 : Moyenne par livre
    results_item = baseline_item_mean(train_df, test_df)
    
    # Comparaison des résultats
    print(f"\n{'='*80}")
    print("COMPARAISON DES BASELINES")
    print(f"{'='*80}")
    print(f"{'Modèle':<30} {'RMSE':<12} {'MAE':<12} {'Temps (s)':<12}")
    print(f"{'-'*80}")
    print(f"{results_global['model']:<30} {results_global['rmse']:<12.4f} {results_global['mae']:<12.4f} {results_global['time_seconds']:<12.4f}")
    print(f"{results_item['model']:<30} {results_item['rmse']:<12.4f} {results_item['mae']:<12.4f} {results_item['time_seconds']:<12.4f}")
    print(f"{'-'*80}")
    
    # Amélioration relative
    rmse_improvement = ((results_global['rmse'] - results_item['rmse']) / results_global['rmse']) * 100
    mae_improvement = ((results_global['mae'] - results_item['mae']) / results_global['mae']) * 100
    
    print(f"\nAmélioration de la moyenne par livre vs moyenne globale:")
    print(f"  RMSE: {rmse_improvement:+.2f}%")
    print(f"  MAE: {mae_improvement:+.2f}%")
    
    # Sauvegarde des résultats
    results_summary = {
        'dataset': output_name,
        'train_size': len(train_df),
        'test_size': len(test_df),
        'n_users': int(train_df['user_id'].nunique()),
        'n_items': int(train_df['parent_asin'].nunique()),
        'baseline_global_mean': results_global,
        'baseline_item_mean': results_item,
        'improvements': {
            'rmse_improvement_percent': rmse_improvement,
            'mae_improvement_percent': mae_improvement
        }
    }
    
    # Sauvegarde en JSON
    output_file = OUTPUT_ROOT / f"baselines_results_{output_name}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nRésultats sauvegardés dans: {output_file}")
    
    return results_summary


# ============================================================================
# EXÉCUTION PRINCIPALE
# ============================================================================

if __name__ == "__main__":
    
    # Évaluation sur le dataset 50k utilisateurs actifs
    print("\n" + "="*80)
    print("ÉVALUATION DES MODÈLES BASELINE")
    print("="*80)
    
    results_50k = evaluate_baselines(
        file_path_50k_train,
        file_path_50k_test,
        "50k_active_users"
    )
    
    # Évaluation sur le dataset temporel (optionnel)
    results_temp = evaluate_baselines(
        file_path_temp_train,
        file_path_temp_test,
        "temporal"
    )
    
    print("\n" + "="*80)
    print("ÉVALUATION TERMINÉE")
    print("="*80)
