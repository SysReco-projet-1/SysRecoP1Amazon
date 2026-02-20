# SysRecoP1Amazon

> **INF6083 — Systèmes de recommandation | Projet P1**  
> Université du Québec en Outaouais — Hiver 2026  
> Enseignant : Étienne Gaël Tajeuna

Implémentation de concepts fondamentaux des systèmes de recommandation collaboratifs sur le dataset **Amazon Reviews 2023 (Books)** — 29,5 millions d'évaluations, 10,3 millions d'utilisateurs, 4,4 millions de livres.

---

## Équipe

| Nom | Rôle |
|-----|------|
| Théo Ambrogiani | Tâche 2 — Graphe biparti |
| Alexandre Boyère | Tâche 0 — Prétraitement |
| Abdelaziz Essafi | Tâche 1 — Similarité |
| Théo Facciotti | Tâches 3 & 4 — Clustering / k-NN |

---

## Structure du projet

```
SysRecoP1Amazon/
│
├── main.py                          # Point d'entrée principal — menu interactif
├── requirements.txt                 # Dépendances Python
│
├── input/                           # Données brutes (non versionnées)
│   ├── amazon_books_sample_active_users.csv
│   └── amazon_books_sample_temporal.csv
│
├── outputs/                         # Résultats générés automatiquement
│   ├── figures/                     # Graphiques et visualisations (.png)
│   ├── similarity/                  # Matrices de similarité et tableaux comparatifs
│   ├── graph/                       # Métriques du graphe biparti
│   ├── matrice/                     # Matrices utilisateur-item
│   ├── mappings/                    # Mappings user_id / item_id
│   └── splits/                      # Ensembles train / test (80/20)
│
├── step_3_1_3_pretraitement/        # Tâche 0 — Prétraitement
│   ├── script_pretraitement.py      # Pipeline de nettoyage et filtrage
│   ├── script_matrix.py             # Construction de la matrice sparse
│   ├── script_sparsity.py           # Analyse de la sparsité
│   └── script_crossvalidation.py   # Séparation train/test
│
└── step_3_2/                        # Tâche 2 — Graphe biparti
    └── script_graph.py              # Construction et analyse du graphe
```

---

## Prérequis

- Python **3.9+**
- pip

---

## Installation

**1. Cloner le dépôt**
```bash
git clone https://github.com/<votre-repo>/SysRecoP1Amazon.git
cd SysRecoP1Amazon
```

**2. Installer les dépendances**
```bash
pip install -r requirements.txt
```

Dépendances principales : `pandas`, `numpy`, `scipy`, `scikit-learn`, `networkx`, `matplotlib`, `seaborn`

**3. Préparer les données**

Extraire les fichiers CSV du zip `3.1.1.zip` dans le dossier `input/` :
```
input/
├── amazon_books_sample_active_users.csv
└── amazon_books_sample_temporal.csv
```

> ⚠️ Les fichiers `.csv`, `.zip` et `.gz` sont exclus du versioning (`.gitignore`). Ne pas les committer.

**4. Lancer le prétraitement (une seule fois)**
```bash
python step_3_1_3_pretraitement/script_pretraitement.py
```

Cette étape génère les fichiers nécessaires dans `outputs/` pour toutes les tâches suivantes.

---

## Lancement

```bash
python main.py
```

Un menu interactif s'affiche :

```
=======================================================
   INF6083 — Systèmes de recommandation — Projet P1
=======================================================
  [0] Tâche 0 - Chargement, échantillonnage et prétraitement
  [1] Tâche 1 - Mesures de similarité
  [2] Tâche 2 - Représentation en graphe biparti
  [3] Tâche 3 - Regroupement des utilisateurs (K-Means)
  [4] Tâche 4 - Prédiction des évaluations (k-NN)
  [a] Exécuter toutes les tâches
  [q] Quitter
=======================================================
```

### Exécution individuelle des scripts

Certaines tâches peuvent aussi être lancées directement :

```bash
# Tâche 2 — Graphe biparti uniquement
python step_3_2/script_graph.py
```

---

## Description des tâches

### Tâche 0 — Prétraitement
- Nettoyage des évaluations invalides (NaN, hors intervalle [1,5])
- Gestion des timestamps aberrants
- Filtrage itératif des utilisateurs et livres sous-représentés
- Construction de la matrice sparse CSR (scipy)
- Séparation train/test (80/20) par utilisateur

### Tâche 1 — Mesures de similarité
- Similarité cosinus (sklearn, optimisé sparse)
- Corrélation de Pearson (calcul par batch)
- Similarité de Jaccard (précalcul des ensembles)
- Analyse comparative sur 5 profils utilisateurs (très actif / moyen / peu actif)
- Distribution des similarités, heatmap, tableau comparatif des voisins

> **Note performance** : sur 3 000 utilisateurs — cosinus : ~0,6s | Jaccard : ~10s | Pearson : ~30–100s  
> Paramètre `MAX_USERS` dans `script_similarity.py` pour ajuster la taille de l'échantillon.

### Tâche 2 — Graphe biparti
- Construction du graphe G = (U ∪ I, E) pondéré par les ratings
- Analyse des métriques : degrés, densité, centralité, clustering, composantes connexes
- Visualisation d'un sous-graphe (30 utilisateurs actifs × 50 livres populaires)

### Tâche 3 — Regroupement K-Means
- Normalisation sparse (StandardScaler sans centrage)
- Sélection du K optimal par score de silhouette (K ∈ [3,8])
- Analyse et caractérisation des clusters
- Visualisation 2D (PCA)

### Tâche 4 — Prédiction k-NN
- Baselines : moyenne globale, moyenne par livre
- k-NN collaboratif avec k ∈ {10, 20, 30, 50, 100}
- Évaluation RMSE / MAE sur l'ensemble de test
- Meilleure configuration : k=100, Pearson (RMSE=0,6310, −34,86% vs baseline)

---

## Outputs générés

Après exécution, les fichiers suivants sont disponibles dans `outputs/` :

| Fichier | Description |
|---------|-------------|
| `figures/distribution_similarites_*.png` | Histogrammes des distributions de similarité |
| `figures/heatmap_similarites_*.png` | Heatmap cosinus (30 users) |
| `figures/subgraph_*.png` | Visualisation du sous-graphe biparti |
| `figures/degree_distribution_*.png` | Distribution des degrés (log-log) |
| `similarity/comparaison_voisins_*.csv` | Tableau comparatif des 10 voisins |
| `similarity/temps_calcul_*.csv` | Temps de calcul par mesure |
| `graph/global_metrics_*.csv` | Métriques globales du graphe |
| `graph/book_centrality_*.csv` | Top 20 livres par centralité |

---

## Notes importantes

- Le paramètre `MAX_USERS = 3000` dans `script_similarity.py` utilise une **sélection stratifiée** (1000 très actifs + 1000 moyens + 1000 peu actifs) pour garantir la représentativité des profils dans l'analyse comparative.
- Mettre `MAX_USERS = None` pour utiliser tous les utilisateurs (attention : temps de calcul très élevé pour Pearson).
- Les fichiers de données volumineuses (`.csv`, `.zip`, `.gz`, `.jsonl`) sont exclus du dépôt Git.