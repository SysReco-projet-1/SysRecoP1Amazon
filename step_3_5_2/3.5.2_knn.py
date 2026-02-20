import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, load_npz
import time
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

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
MATRIX = Path("outputs") / "matrices"
MAPPINGS = Path("outputs") / "mappings"

file_path_50k_train = PROJECT_ROOT / SPLITS / "train_amazon_books_sample_active_users.csv"
file_path_50k_test = PROJECT_ROOT / SPLITS / "test_amazon_books_sample_active_users.csv"

# Chemins vers les matrices et mappings
matrix_train_path = PROJECT_ROOT / MATRIX / "mat_csr_amazon_books_sample_active_users.npz"
user_mapping_path = PROJECT_ROOT / MAPPINGS / "user_mapping_amazon_books_sample_active_users.csv"
item_mapping_path = PROJECT_ROOT / MAPPINGS / "item_mapping_amazon_books_sample_active_users.csv"

# ============================================================================
# 1. Calcul des mesures de similarité
# — Similarité cosinus
# — Corrélation de Pearson
# — Similarité de Jaccard
# ============================================================================

def compute_cosine_similarity(R_train):
    """
    Calcule la similarité cosinus entre utilisateurs
    
    Args:
        R_train: Matrice sparse CSR (users x items)
    
    Returns:
        Matrice de similarité (users x users)
    """
    print("Calcul de la similarité cosinus...")
    start_time = time.time()
    
    # sklearn gère efficacement les matrices sparse
    sim_matrix = cosine_similarity(R_train, dense_output=False)
    
    # Mettre la diagonale à 0 (un utilisateur n'est pas son propre voisin)
    sim_matrix.setdiag(0)
    
    elapsed = time.time() - start_time
    print(f"  Temps: {elapsed:.2f}s")
    
    return sim_matrix


def compute_pearson_similarity(R_train):
    """
    Calcule la corrélation de Pearson entre utilisateurs
    
    Args:
        R_train: Matrice sparse CSR (users x items)
    
    Returns:
        Matrice de similarité (users x users)
    """
    print("Calcul de la corrélation de Pearson...")
    start_time = time.time()
    
    # Convertir en dense pour le calcul (attention à la mémoire)
    # Pour très grandes matrices, faire par batch
    n_users = R_train.shape[0]
    
    # Calculer les moyennes par utilisateur (seulement sur les items notés)
    user_means = np.array(R_train.sum(axis=1) / (R_train.getnnz(axis=1)[:, None] + 1e-9)).flatten()
    
    # Centrer les données
    R_centered = R_train.copy()
    for i in range(n_users):
        indices = R_train[i].indices
        R_centered.data[R_centered.indptr[i]:R_centered.indptr[i+1]] -= user_means[i]
    
    # Calculer la corrélation (similaire au cosinus sur données centrées)
    sim_matrix = cosine_similarity(R_centered, dense_output=False)
    sim_matrix.setdiag(0)
    
    elapsed = time.time() - start_time
    print(f"  Temps: {elapsed:.2f}s")
    
    return sim_matrix


def compute_jaccard_similarity(R_train):
    """
    Calcule la similarité de Jaccard entre utilisateurs
    (basée sur les items en commun, pas les ratings)
    
    Args:
        R_train: Matrice sparse CSR (users x items)
    
    Returns:
        Matrice de similarité (users x users)
    """
    print("Calcul de la similarité de Jaccard...")
    start_time = time.time()
    
    # Binariser la matrice (1 si noté, 0 sinon)
    R_binary = (R_train > 0).astype(float)
    
    # Intersection: nombre d'items en commun
    intersection = R_binary.dot(R_binary.T)
    
    # Union: |A| + |B| - |A ∩ B|
    row_sums = np.array(R_binary.sum(axis=1)).flatten()
    union = row_sums[:, None] + row_sums[None, :] - intersection.toarray()
    
    # Jaccard = intersection / union
    with np.errstate(divide='ignore', invalid='ignore'):
        sim_matrix = intersection.toarray() / union
        sim_matrix[np.isnan(sim_matrix)] = 0
    
    # Mettre la diagonale à 0
    np.fill_diagonal(sim_matrix, 0)
    
    elapsed = time.time() - start_time
    print(f"  Temps: {elapsed:.2f}s")
    
    return sim_matrix


# ============================================================================
# 2. Pré-calcul des k plus proches voisins
# — Pour chaque utilisateur, stocker ses k voisins les plus similaires
# — Optimisation: évite de recalculer à chaque prédiction
# ============================================================================

def precompute_top_k_neighbors(sim_matrix, k):
    """
    Pré-calcule les k plus proches voisins pour chaque utilisateur
    
    Args:
        sim_matrix: Matrice de similarité (users x users)
        k: Nombre de voisins à conserver
    
    Returns:
        dict: {user_idx: [(neighbor_idx, similarity), ...]}
    """
    print(f"Pré-calcul des top-{k} voisins...")
    start_time = time.time()
    
    n_users = sim_matrix.shape[0]
    top_k_neighbors = {}
    
    # Convertir en array si sparse
    if hasattr(sim_matrix, 'toarray'):
        sim_array = sim_matrix.toarray()
    else:
        sim_array = sim_matrix
    
    for user_idx in range(n_users):
        # Obtenir les similarités pour cet utilisateur
        similarities = sim_array[user_idx]
        
        # Trouver les k plus grands indices (en excluant l'utilisateur lui-même)
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
        
        # Stocker les voisins avec leurs similarités
        neighbors = [(int(idx), float(similarities[idx])) for idx in top_k_indices if similarities[idx] > 0]
        top_k_neighbors[user_idx] = neighbors
    
    elapsed = time.time() - start_time
    print(f"  Temps: {elapsed:.2f}s")
    
    return top_k_neighbors


# ============================================================================
# 3. Prédiction k-NN
# — Formule: r̂_u,i = r̄_u + Σ(sim(u,v) * (r_v,i - r̄_v)) / Σ|sim(u,v)|
# — Gestion des cas limites
# ============================================================================

def predict_knn(user_idx, item_idx, R_train, user_means, top_k_neighbors, global_mean, item_means):
    """
    Prédit le rating d'un utilisateur pour un item avec k-NN
    
    Args:
        user_idx: Index de l'utilisateur
        item_idx: Index de l'item
        R_train: Matrice d'entraînement sparse
        user_means: Moyennes par utilisateur
        top_k_neighbors: Dictionnaire des k voisins pré-calculés
        global_mean: Moyenne globale (fallback)
        item_means: Moyennes par item (fallback)
    
    Returns:
        Prédiction du rating
    """
    # Cas 1: Utilisateur inconnu
    if user_idx >= len(user_means):
        return item_means.get(item_idx, global_mean)
    
    user_mean = user_means[user_idx]
    
    # Obtenir les voisins de l'utilisateur
    neighbors = top_k_neighbors.get(user_idx, [])
    
    # Filtrer les voisins qui ont noté cet item
    relevant_neighbors = []
    for neighbor_idx, similarity in neighbors:
        if R_train[neighbor_idx, item_idx] != 0:
            relevant_neighbors.append((neighbor_idx, similarity))
    
    # Cas 2: Aucun voisin n'a noté cet item
    if len(relevant_neighbors) == 0:
        # Fallback: moyenne de l'utilisateur ou moyenne de l'item
        if user_mean > 0:
            return user_mean
        else:
            return item_means.get(item_idx, global_mean)
    
    # Cas 3: Prédiction k-NN standard
    numerator = 0.0
    denominator = 0.0
    
    for neighbor_idx, similarity in relevant_neighbors:
        neighbor_rating = R_train[neighbor_idx, item_idx]
        neighbor_mean = user_means[neighbor_idx]
        
        numerator += similarity * (neighbor_rating - neighbor_mean)
        denominator += abs(similarity)
    
    if denominator == 0:
        return user_mean if user_mean > 0 else global_mean
    
    prediction = user_mean + (numerator / denominator)
    
    # Clipper entre 1 et 5
    return np.clip(prediction, 1.0, 5.0)


# ============================================================================
# 4. Évaluation k-NN sur l'ensemble de test
# — Teste différentes valeurs de k et mesures de similarité
# — Calcule RMSE, MAE et temps d'exécution
# ============================================================================

def evaluate_knn(R_train, test_df, user_mapping, item_mapping, k, similarity_type, 
                 top_k_neighbors, user_means, global_mean, item_means, sample_ratio=1.0):
    """
    Évalue le modèle k-NN sur l'ensemble de test
    
    Args:
        R_train: Matrice d'entraînement
        test_df: DataFrame de test
        user_mapping: Mapping user_id -> user_idx
        item_mapping: Mapping parent_asin -> item_idx
        k: Nombre de voisins
        similarity_type: Type de similarité utilisé
        top_k_neighbors: Voisins pré-calculés
        user_means: Moyennes par utilisateur
        global_mean: Moyenne globale
        item_means: Moyennes par item
        sample_ratio: Ratio d'échantillonnage du test set (pour accélérer)
    
    Returns:
        dict: Résultats (RMSE, MAE, temps)
    """
    print(f"\nÉvaluation k-NN (k={k}, similarité={similarity_type})...")
    
    # Échantillonner le test set si demandé
    if sample_ratio < 1.0:
        test_sample = test_df.sample(frac=sample_ratio, random_state=42)
        print(f"  Échantillon de test: {len(test_sample)} évaluations ({sample_ratio*100:.0f}%)")
    else:
        test_sample = test_df
    
    start_time = time.time()
    
    predictions = []
    true_ratings = []
    
    # Créer les mappings inverses
    user_id_to_idx = {uid: idx for idx, uid in user_mapping.items()}
    item_id_to_idx = {iid: idx for idx, iid in item_mapping.items()}
    
    for _, row in test_sample.iterrows():
        user_id = row['user_id']
        item_id = row['parent_asin']
        true_rating = row['rating']
        
        # Obtenir les indices
        user_idx = user_id_to_idx.get(user_id, -1)
        item_idx = item_id_to_idx.get(item_id, -1)
        
        # Prédire
        if user_idx == -1 or item_idx == -1:
            # Utilisateur ou item inconnu
            pred = item_means.get(item_idx, global_mean)
        else:
            pred = predict_knn(user_idx, item_idx, R_train, user_means, 
                             top_k_neighbors, global_mean, item_means)
        
        predictions.append(pred)
        true_ratings.append(true_rating)
    
    predictions = np.array(predictions)
    true_ratings = np.array(true_ratings)
    
    # Calculer les métriques
    rmse = np.sqrt(mean_squared_error(true_ratings, predictions))
    mae = mean_absolute_error(true_ratings, predictions)
    
    elapsed = time.time() - start_time
    
    results = {
        'model': f'k-NN (k={k}, {similarity_type})',
        'k': k,
        'similarity': similarity_type,
        'rmse': rmse,
        'mae': mae,
        'time_seconds': elapsed,
        'n_predictions': len(test_sample)
    }
    
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Temps: {elapsed:.2f}s")
    
    return results


# ============================================================================
# 5. Fonction principale pour optimiser les hyperparamètres
# — Teste différentes valeurs de k: {10, 20, 30, 50, 100}
# — Teste différentes mesures: cosinus, Pearson, Jaccard
# — Identifie la meilleure configuration
# ============================================================================

def optimize_knn_hyperparameters(file_path_train, file_path_test, output_name, 
                                  k_values=[10, 20, 30, 50, 100],
                                  test_sample_ratio=0.2):
    """
    Optimise les hyperparamètres du k-NN
    
    Args:
        file_path_train: Chemin vers le CSV d'entraînement
        file_path_test: Chemin vers le CSV de test
        output_name: Nom pour les fichiers de sortie
        k_values: Liste des valeurs de k à tester
        test_sample_ratio: Ratio d'échantillonnage pour accélérer
    """
    print(f"\n{'='*80}")
    print(f"OPTIMISATION k-NN: {output_name}")
    print(f"{'='*80}\n")
    
    # Chargement des données
    print("Chargement des données...")
    train_df = pd.read_csv(file_path_train)
    test_df = pd.read_csv(file_path_test)
    
    # Charger la matrice et les mappings
    R_train = load_npz(matrix_train_path)
    user_mapping = pd.read_csv(user_mapping_path, index_col=0).squeeze().to_dict()
    item_mapping = pd.read_csv(item_mapping_path, index_col=0).squeeze().to_dict()
    
    print(f"Matrice: {R_train.shape} (users x items)")
    print(f"Test set: {len(test_df)} évaluations")
    print(f"Échantillonnage du test: {test_sample_ratio*100:.0f}%\n")
    
    # Calculer les moyennes (pour les prédictions)
    global_mean = train_df['rating'].mean()
    user_means = np.array(R_train.sum(axis=1) / (R_train.getnnz(axis=1)[:, None] + 1e-9)).flatten()
    item_means = train_df.groupby('parent_asin')['rating'].mean().to_dict()
    
    # Calculer les matrices de similarité
    print("\n" + "="*80)
    print("CALCUL DES MATRICES DE SIMILARITÉ")
    print("="*80 + "\n")
    
    similarities = {
        'cosinus': compute_cosine_similarity(R_train),
        'pearson': compute_pearson_similarity(R_train),
        'jaccard': compute_jaccard_similarity(R_train)
    }
    
    # Tester toutes les configurations
    print("\n" + "="*80)
    print("ÉVALUATION DES CONFIGURATIONS")
    print("="*80)
    
    all_results = []
    
    for sim_name, sim_matrix in similarities.items():
        print(f"\n--- Similarité: {sim_name} ---")
        
        for k in k_values:
            # Pré-calculer les voisins
            top_k_neighbors = precompute_top_k_neighbors(sim_matrix, k)
            
            # Évaluer
            results = evaluate_knn(
                R_train, test_df, user_mapping, item_mapping,
                k, sim_name, top_k_neighbors, user_means,
                global_mean, item_means, sample_ratio=test_sample_ratio
            )
            
            all_results.append(results)
    
    # Identifier la meilleure configuration
    best_result = min(all_results, key=lambda x: x['rmse'])
    
    print("\n" + "="*80)
    print("MEILLEURE CONFIGURATION")
    print("="*80)
    print(f"k = {best_result['k']}")
    print(f"Similarité = {best_result['similarity']}")
    print(f"RMSE = {best_result['rmse']:.4f}")
    print(f"MAE = {best_result['mae']:.4f}")
    
    # Sauvegarder les résultats
    summary = {
        'dataset': output_name,
        'train_size': len(train_df),
        'test_size': len(test_df),
        'test_sample_ratio': test_sample_ratio,
        'k_values_tested': k_values,
        'similarities_tested': list(similarities.keys()),
        'all_results': all_results,
        'best_configuration': best_result
    }
    
    output_file = OUTPUT_ROOT / f"knn_optimization_{output_name}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nRésultats sauvegardés: {output_file}")
    
    return summary


# ============================================================================
# EXÉCUTION PRINCIPALE
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("OPTIMISATION DES HYPERPARAMÈTRES k-NN")
    print("="*80)
    
    # Optimisation sur le dataset 50k utilisateurs actifs
    results = optimize_knn_hyperparameters(
        file_path_50k_train,
        file_path_50k_test,
        "50k_active_users",
        k_values=[10, 20, 30, 50, 100],
        test_sample_ratio=0.2  # 20% du test set pour accélérer
    )
    
    print("\n" + "="*80)
    print("OPTIMISATION TERMINÉE")
    print("="*80)
