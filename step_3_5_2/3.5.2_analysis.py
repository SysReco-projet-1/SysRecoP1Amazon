import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# Variables de l'arboréscence du projet
# =====================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = Path(__file__).resolve().parent
FILE_NAME = Path(__file__).resolve().stem

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / TASK_ROOT.name / "3.5.2_knn"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Chemins vers les résultats
knn_results_path = OUTPUT_ROOT / "knn_optimization_50k_active_users.json"
baseline_results_path = PROJECT_ROOT / "outputs" / "step_3_5_1" / "3.5.1_baselines" / "baselines_results_50k_active_users.json"

# Configuration des graphiques
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# ============================================================================
# 1. Chargement des résultats
# ============================================================================

def load_results():
    """
    Charge les résultats k-NN et baselines
    """
    print("Chargement des résultats...")
    
    with open(knn_results_path, 'r', encoding='utf-8') as f:
        knn_results = json.load(f)
    
    with open(baseline_results_path, 'r', encoding='utf-8') as f:
        baseline_results = json.load(f)
    
    print(f"  k-NN: {len(knn_results['all_results'])} configurations")
    print(f"  Baselines: 2 modèles")
    
    return knn_results, baseline_results


# ============================================================================
# 2. Tableau comparatif complet
# ============================================================================

def create_comparison_table(knn_results, baseline_results):
    """
    Crée un tableau comparant tous les modèles
    """
    print("\n" + "="*80)
    print("TABLEAU COMPARATIF DES MODÈLES")
    print("="*80 + "\n")
    
    # Préparer les données
    rows = []
    
    # Baselines
    rows.append({
        'Modèle': 'Baseline - Moyenne globale',
        'RMSE': baseline_results['baseline_global_mean']['rmse'],
        'MAE': baseline_results['baseline_global_mean']['mae'],
        'Temps (s)': baseline_results['baseline_global_mean']['time_seconds']
    })
    
    rows.append({
        'Modèle': 'Baseline - Moyenne par livre',
        'RMSE': baseline_results['baseline_item_mean']['rmse'],
        'MAE': baseline_results['baseline_item_mean']['mae'],
        'Temps (s)': baseline_results['baseline_item_mean']['time_seconds']
    })
    
    # k-NN: sélectionner quelques configurations représentatives
    all_knn = knn_results['all_results']
    
    # Trouver des configurations spécifiques
    for result in all_knn:
        if result['k'] == 10 and result['similarity'] == 'cosinus':
            rows.append({
                'Modèle': f"k-NN (k={result['k']}, {result['similarity']})",
                'RMSE': result['rmse'],
                'MAE': result['mae'],
                'Temps (s)': result['time_seconds']
            })
        elif result['k'] == 20 and result['similarity'] == 'pearson':
            rows.append({
                'Modèle': f"k-NN (k={result['k']}, {result['similarity']})",
                'RMSE': result['rmse'],
                'MAE': result['mae'],
                'Temps (s)': result['time_seconds']
            })
    
    # Meilleure configuration
    best = knn_results['best_configuration']
    rows.append({
        'Modèle': f"k-NN (k={best['k']}, {best['similarity']}) ★ MEILLEUR",
        'RMSE': best['rmse'],
        'MAE': best['mae'],
        'Temps (s)': best['time_seconds']
    })
    
    # Créer le DataFrame
    df = pd.DataFrame(rows)
    
    # Afficher le tableau
    print(df.to_string(index=False))
    
    # Sauvegarder en CSV
    output_file = OUTPUT_ROOT / "comparison_table.csv"
    df.to_csv(output_file, index=False)
    print(f"\nTableau sauvegardé: {output_file}")
    
    return df


# ============================================================================
# 3. Graphique: RMSE vs k pour chaque similarité
# ============================================================================

def plot_rmse_vs_k(knn_results):
    """
    Graphique montrant l'évolution du RMSE en fonction de k
    """
    print("\nGénération du graphique RMSE vs k...")
    
    all_results = knn_results['all_results']
    
    # Organiser les données par similarité
    data = {}
    for result in all_results:
        sim = result['similarity']
        if sim not in data:
            data[sim] = {'k': [], 'rmse': []}
        data[sim]['k'].append(result['k'])
        data[sim]['rmse'].append(result['rmse'])
    
    # Créer le graphique
    plt.figure(figsize=(10, 6))
    
    colors = {'cosinus': '#1f77b4', 'pearson': '#ff7f0e', 'jaccard': '#2ca02c'}
    markers = {'cosinus': 'o', 'pearson': 's', 'jaccard': '^'}
    
    for sim, values in data.items():
        # Trier par k
        sorted_indices = np.argsort(values['k'])
        k_sorted = [values['k'][i] for i in sorted_indices]
        rmse_sorted = [values['rmse'][i] for i in sorted_indices]
        
        plt.plot(k_sorted, rmse_sorted, marker=markers[sim], 
                label=sim.capitalize(), linewidth=2, markersize=8,
                color=colors[sim])
    
    plt.xlabel('Nombre de voisins (k)', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('Impact de k sur le RMSE pour différentes mesures de similarité', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = OUTPUT_ROOT / "knn_rmse_vs_k.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Sauvegardé: {output_file}")


# ============================================================================
# 4. Graphique: MAE vs k pour chaque similarité
# ============================================================================

def plot_mae_vs_k(knn_results):
    """
    Graphique montrant l'évolution du MAE en fonction de k
    """
    print("Génération du graphique MAE vs k...")
    
    all_results = knn_results['all_results']
    
    # Organiser les données par similarité
    data = {}
    for result in all_results:
        sim = result['similarity']
        if sim not in data:
            data[sim] = {'k': [], 'mae': []}
        data[sim]['k'].append(result['k'])
        data[sim]['mae'].append(result['mae'])
    
    # Créer le graphique
    plt.figure(figsize=(10, 6))
    
    colors = {'cosinus': '#1f77b4', 'pearson': '#ff7f0e', 'jaccard': '#2ca02c'}
    markers = {'cosinus': 'o', 'pearson': 's', 'jaccard': '^'}
    
    for sim, values in data.items():
        # Trier par k
        sorted_indices = np.argsort(values['k'])
        k_sorted = [values['k'][i] for i in sorted_indices]
        mae_sorted = [values['mae'][i] for i in sorted_indices]
        
        plt.plot(k_sorted, mae_sorted, marker=markers[sim], 
                label=sim.capitalize(), linewidth=2, markersize=8,
                color=colors[sim])
    
    plt.xlabel('Nombre de voisins (k)', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('Impact de k sur le MAE pour différentes mesures de similarité', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = OUTPUT_ROOT / "knn_mae_vs_k.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Sauvegardé: {output_file}")


# ============================================================================
# 5. Graphique: Comparaison k-NN vs Baselines
# ============================================================================

def plot_knn_vs_baselines(knn_results, baseline_results):
    """
    Compare les meilleures configurations k-NN avec les baselines
    """
    print("Génération du graphique k-NN vs Baselines...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Préparer les données
    models = [
        'Moyenne\nglobale',
        'Moyenne\npar livre',
        f"k-NN\n(meilleur)"
    ]
    
    rmse_values = [
        baseline_results['baseline_global_mean']['rmse'],
        baseline_results['baseline_item_mean']['rmse'],
        knn_results['best_configuration']['rmse']
    ]
    
    mae_values = [
        baseline_results['baseline_global_mean']['mae'],
        baseline_results['baseline_item_mean']['mae'],
        knn_results['best_configuration']['mae']
    ]
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    
    # RMSE
    bars1 = axes[0].bar(models, rmse_values, color=colors, alpha=0.8)
    axes[0].set_ylabel('RMSE', fontsize=12)
    axes[0].set_title('Comparaison RMSE', fontsize=13)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for bar, value in zip(bars1, rmse_values):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=10)
    
    # MAE
    bars2 = axes[1].bar(models, mae_values, color=colors, alpha=0.8)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Comparaison MAE', fontsize=13)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for bar, value in zip(bars2, mae_values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    output_file = OUTPUT_ROOT / "knn_vs_baselines.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Sauvegardé: {output_file}")


# ============================================================================
# 6. Graphique: Temps d'exécution vs k
# ============================================================================

def plot_time_vs_k(knn_results):
    """
    Graphique montrant le temps d'exécution en fonction de k
    """
    print("Génération du graphique Temps vs k...")
    
    all_results = knn_results['all_results']
    
    # Organiser les données par similarité
    data = {}
    for result in all_results:
        sim = result['similarity']
        if sim not in data:
            data[sim] = {'k': [], 'time': []}
        data[sim]['k'].append(result['k'])
        data[sim]['time'].append(result['time_seconds'])
    
    # Créer le graphique
    plt.figure(figsize=(10, 6))
    
    colors = {'cosinus': '#1f77b4', 'pearson': '#ff7f0e', 'jaccard': '#2ca02c'}
    markers = {'cosinus': 'o', 'pearson': 's', 'jaccard': '^'}
    
    for sim, values in data.items():
        # Trier par k
        sorted_indices = np.argsort(values['k'])
        k_sorted = [values['k'][i] for i in sorted_indices]
        time_sorted = [values['time'][i] for i in sorted_indices]
        
        plt.plot(k_sorted, time_sorted, marker=markers[sim], 
                label=sim.capitalize(), linewidth=2, markersize=8,
                color=colors[sim])
    
    plt.xlabel('Nombre de voisins (k)', fontsize=12)
    plt.ylabel('Temps d\'exécution (secondes)', fontsize=12)
    plt.title('Impact de k sur le temps d\'exécution', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = OUTPUT_ROOT / "knn_time_vs_k.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Sauvegardé: {output_file}")


# ============================================================================
# 7. Analyse des améliorations
# ============================================================================

def analyze_improvements(knn_results, baseline_results):
    """
    Analyse les améliorations du k-NN par rapport aux baselines
    """
    print("\n" + "="*80)
    print("ANALYSE DES AMÉLIORATIONS")
    print("="*80 + "\n")
    
    best_knn = knn_results['best_configuration']
    baseline_item = baseline_results['baseline_item_mean']
    baseline_global = baseline_results['baseline_global_mean']
    
    # Amélioration vs moyenne par livre
    rmse_improvement_item = ((baseline_item['rmse'] - best_knn['rmse']) / baseline_item['rmse']) * 100
    mae_improvement_item = ((baseline_item['mae'] - best_knn['mae']) / baseline_item['mae']) * 100
    
    # Amélioration vs moyenne globale
    rmse_improvement_global = ((baseline_global['rmse'] - best_knn['rmse']) / baseline_global['rmse']) * 100
    mae_improvement_global = ((baseline_global['mae'] - best_knn['mae']) / baseline_global['mae']) * 100
    
    print(f"Meilleure configuration k-NN:")
    print(f"  k = {best_knn['k']}")
    print(f"  Similarité = {best_knn['similarity']}")
    print(f"  RMSE = {best_knn['rmse']:.4f}")
    print(f"  MAE = {best_knn['mae']:.4f}")
    print()
    
    print(f"Amélioration vs Baseline 'Moyenne par livre':")
    print(f"  RMSE: {rmse_improvement_item:+.2f}%")
    print(f"  MAE: {mae_improvement_item:+.2f}%")
    print()
    
    print(f"Amélioration vs Baseline 'Moyenne globale':")
    print(f"  RMSE: {rmse_improvement_global:+.2f}%")
    print(f"  MAE: {mae_improvement_global:+.2f}%")
    
    # Sauvegarder l'analyse
    analysis = {
        'best_knn': best_knn,
        'improvements_vs_item_baseline': {
            'rmse_percent': rmse_improvement_item,
            'mae_percent': mae_improvement_item
        },
        'improvements_vs_global_baseline': {
            'rmse_percent': rmse_improvement_global,
            'mae_percent': mae_improvement_global
        }
    }
    
    output_file = OUTPUT_ROOT / "improvements_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"\nAnalyse sauvegardée: {output_file}")


# ============================================================================
# 8. Comparaison des mesures de similarité
# ============================================================================

def compare_similarities(knn_results):
    """
    Compare les performances des différentes mesures de similarité
    """
    print("\n" + "="*80)
    print("COMPARAISON DES MESURES DE SIMILARITÉ")
    print("="*80 + "\n")
    
    all_results = knn_results['all_results']
    
    # Calculer les moyennes par similarité
    sim_stats = {}
    for result in all_results:
        sim = result['similarity']
        if sim not in sim_stats:
            sim_stats[sim] = {'rmse': [], 'mae': [], 'time': []}
        sim_stats[sim]['rmse'].append(result['rmse'])
        sim_stats[sim]['mae'].append(result['mae'])
        sim_stats[sim]['time'].append(result['time_seconds'])
    
    # Afficher les statistiques
    print(f"{'Similarité':<15} {'RMSE moyen':<15} {'MAE moyen':<15} {'Temps moyen (s)':<15}")
    print("-" * 60)
    
    for sim, stats in sim_stats.items():
        avg_rmse = np.mean(stats['rmse'])
        avg_mae = np.mean(stats['mae'])
        avg_time = np.mean(stats['time'])
        print(f"{sim.capitalize():<15} {avg_rmse:<15.4f} {avg_mae:<15.4f} {avg_time:<15.2f}")
    
    # Graphique de comparaison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sims = list(sim_stats.keys())
    avg_rmse = [np.mean(sim_stats[s]['rmse']) for s in sims]
    avg_mae = [np.mean(sim_stats[s]['mae']) for s in sims]
    avg_time = [np.mean(sim_stats[s]['time']) for s in sims]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    axes[0].bar(sims, avg_rmse, color=colors, alpha=0.8)
    axes[0].set_ylabel('RMSE moyen')
    axes[0].set_title('RMSE par similarité')
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].bar(sims, avg_mae, color=colors, alpha=0.8)
    axes[1].set_ylabel('MAE moyen')
    axes[1].set_title('MAE par similarité')
    axes[1].grid(axis='y', alpha=0.3)
    
    axes[2].bar(sims, avg_time, color=colors, alpha=0.8)
    axes[2].set_ylabel('Temps moyen (s)')
    axes[2].set_title('Temps par similarité')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_file = OUTPUT_ROOT / "similarity_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nGraphique sauvegardé: {output_file}")


# ============================================================================
# EXÉCUTION PRINCIPALE
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("ANALYSE DES RÉSULTATS k-NN")
    print("="*80)
    
    # Charger les résultats
    knn_results, baseline_results = load_results()
    
    # Créer le tableau comparatif
    create_comparison_table(knn_results, baseline_results)
    
    # Générer les graphiques
    print("\n" + "="*80)
    print("GÉNÉRATION DES GRAPHIQUES")
    print("="*80)
    
    plot_rmse_vs_k(knn_results)
    plot_mae_vs_k(knn_results)
    plot_time_vs_k(knn_results)
    plot_knn_vs_baselines(knn_results, baseline_results)
    
    # Analyses
    analyze_improvements(knn_results, baseline_results)
    compare_similarities(knn_results)
    
    print("\n" + "="*80)
    print("ANALYSE TERMINÉE")
    print("="*80)
    print(f"\nTous les fichiers sont dans: {OUTPUT_ROOT}")
