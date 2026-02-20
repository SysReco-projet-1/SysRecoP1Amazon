import networkx as nx
import scipy.sparse as sp
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
from scipy.sparse.csgraph import connected_components

# ============================================================
# Gestion des chemins du projet
# ============================================================

# ROOT = dossier racine du projet
ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input"

# Dossier contenant les matrices et schémas
OUTPUT = ROOT / "outputs"
MATRIX = OUTPUT / "matrice"
FIGURES = OUTPUT / "figures"
FIGURES2 = OUTPUT / "figures_step_3_3_1"

# Dossier contenant les mappings (user_idx -> user_id, item_idx -> parent_asin)
MAPPINGS = OUTPUT / "mappings"

FIGURES2.mkdir(parents=True, exist_ok=True)


# ============================================================
# Fonction principale : génération du graphe biparti
# ============================================================

def bipartite_graph_gen(csvfile, matrixfile):
    """
    Génère un graphe biparti utilisateur-livre à partir :
    - d'une matrice sparse CSR nomée R
    - des fichiers de mapping user/item
    
    Construit :
    - le graphe complet
    - un sous-graphe réduit pour visualisation
    """

    # Chargement de la matrice sparse
    R = load_mat(matrixfile)

    # Création du graphe complet à partir de la matrice et des mappings
    G, user_ids, item_ids = create_graph(R, csvfile)

    # Création du sous-graphe pour visualisation
    subgraph = create_subgraph(G, R, user_ids, item_ids)

    # Crée une visualisation du sous-graphe dans le dossier outputs/figures_step_3_3_1
    save_subgraph(subgraph, csvfile)

    # Analyse du sous-graphe
    analyze_graph(subgraph, R, csvfile)


def load_mat(file):
    """
    Charge une matrice sparse au format CSR.
    Format CSR :
    """
    return sp.load_npz(MATRIX / file)


def create_graph(R, csvfile):
    """
    Construit le graphe biparti utilisateur–livre à partir
    d’une matrice sparse CSR et des fichiers de mapping.
    """

    # Chargement des mappings (index matrice -> id réels)
    user_map = pd.read_csv(MAPPINGS / f"user_mapping_{csvfile}")
    item_map = pd.read_csv(MAPPINGS / f"item_mapping_{csvfile}")

    # Tri pour assurer la correspondance avec les lignes/colonnes de R
    user_map = user_map.sort_values("user_idx")
    item_map = item_map.sort_values("item_idx")

    # Extraction des identifiants
    user_ids = user_map["user_id"].to_numpy()
    item_ids = item_map["parent_asin"].to_numpy()

    # Vérification cohérence dimensions matrice vs mapping
    assert len(user_ids) == R.shape[0]
    assert len(item_ids) == R.shape[1]

    # Création du graphe biparti non orienté
    G = nx.Graph()

    # Ajout des nœuds (partition 0 = utilisateurs, 1 = livres)
    G.add_nodes_from(user_ids, bipartite=0)
    G.add_nodes_from(item_ids, bipartite=1)

    # Ajout des arêtes à partir des valeurs non nulles de la CSR
    # (une arête par interaction user-item)
    G = calc_edges(R, user_ids, item_ids, G)

    return G, user_ids, item_ids


def calc_edges(R, user_ids, item_ids, G):
    """
    Parcourt la matrice CSR ligne par ligne.
    Pour chaque valeur non nulle ru,i :
        - ajoute une arête (u, i)
        - poids = rating
    """

    # Pour chaque utilisateur (ligne de la matrice)
    for u_idx in range(R.shape[0]):

        # Début et fin des éléments non nuls de la ligne u_idx
        start = R.indptr[u_idx]
        end = R.indptr[u_idx + 1]

        # Parcours des interactions non nulles
        for k in range(start, end):

            i_idx = R.indices[k]     # index du livre
            r = float(R.data[k])     # rating

            # Récupération des identifiants réels
            u = user_ids[u_idx]
            i = item_ids[i_idx]

            # Ajout de l’arête pondérée
            G.add_edge(u, i, weight=r)

    return G


def create_subgraph(G, R, user_ids, item_ids):
    """
    Crée un sous-graphe contenant :
        - 30 utilisateurs les plus actifs
        - 50 livres les plus populaires
        - les arêtes correspondantes
    """

    # Degré utilisateur = nombre d'interactions par ligne
    user_degrees = np.diff(R.indptr)

    user_degree_dict = {
        user_ids[u_idx]: int(user_degrees[u_idx])
        for u_idx in range(len(user_ids))
    }

    # Degré item = nombre d’interactions par colonne
    item_degrees = np.array(R.getnnz(axis=0)).flatten()

    item_degree_dict = {
        item_ids[i_idx]: int(item_degrees[i_idx])
        for i_idx in range(len(item_ids))
    }

    # Sélection des top utilisateurs
    top_users = sorted(user_degree_dict,
                       key=user_degree_dict.get,
                       reverse=True)[:30]

    # Sélection des top livres
    top_items = sorted(item_degree_dict,
                       key=item_degree_dict.get,
                       reverse=True)[:50]

    # Sous-ensemble de nœuds
    nodes_subset = top_users + top_items

    # Sous-graphe induit
    G_sub = G.subgraph(nodes_subset)

    return G_sub


def save_subgraph(G_sub, file):
    """
    Visualise le sous-graphe :
        - layout biparti
        - taille des nœuds proportionnelle au degré
        - couleurs différentes pour chaque partition
    """

    # Séparation des partitions
    users = [n for n, d in G_sub.nodes(data=True)
             if d["bipartite"] == 0]

    items = [n for n, d in G_sub.nodes(data=True)
             if d["bipartite"] == 1]

    # Positionnement biparti
    pos = nx.bipartite_layout(G_sub, users)

    # Calcul des degrés
    degrees = dict(G_sub.degree())

    # Taille proportionnelle au degré
    user_sizes = [100 + 40 * degrees[n] for n in users]
    item_sizes = [100 + 40 * degrees[n] for n in items]

    plt.figure(figsize=(14, 8))

    # Nœuds utilisateurs
    nx.draw_networkx_nodes(G_sub, pos,
                           nodelist=users,
                           node_size=user_sizes,
                           node_color="tab:blue",
                           alpha=0.8)

    # Nœuds livres
    nx.draw_networkx_nodes(G_sub, pos,
                           nodelist=items,
                           node_size=item_sizes,
                           node_color="tab:orange",
                           alpha=0.8)

    # Arêtes
    nx.draw_networkx_edges(G_sub, pos,
                           alpha=0.3,
                           width=0.5)

    plt.title("Sous-graphe biparti (30 utilisateurs, 50 livres)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGURES2 / f"subgraph_{Path(file).stem}.png", dpi=300)
    plt.close()


# ==================================
# Analyse du graphe
# ==================================

def analyze_graph(G_sub, R, dataset_name):

    print("\n[GLOBAL METRICS]")
    m = global_metrics_from_R(R)

    print(f"  - |V| (nodes)        : {m['|V|']:,}")
    print(f"  - |E| (edges)        : {m['|E|']:,}")
    print(f"  - Average degree     : {m['d_avg']:.4f}")
    print(f"  - Density            : {m['density']:.8f}")

    print("\n[ITEM CENTRALITY - Top 20 (subgraph)]")
    top20, item_deg = top_item_centrality_subgraph(G_sub)

    for asin, cent in top20:
        print(f"  - {asin}  |  centrality={cent:.4f}  |  degree={item_deg[asin]}")

    print("\n[BIPARTITE CLUSTERING - Subgraph]")
    avg_c, _ = bipartite_clustering_subgraph(G_sub)
    print(f"  - Average clustering coefficient : {avg_c:.6f}")

    print("\n[CONNECTED COMPONENTS - Full Graph]")
    n_comp, max_size = connected_components_from_R(R)
    print(f"  - Number of components : {n_comp}")
    print(f"  - Largest component    : {max_size:,} nodes")


def global_metrics_from_R(R):
    nU, nI = R.shape
    E = R.nnz
    V = nU + nI
    avg_d = 2 * E / V
    density = E / (nU * nI)
    return {
        "|V|": V, "|E|": E,
        "d_avg": avg_d,
        "density": density,
    }


def log_log_degree_distributions(R, file):
    user_deg = np.diff(R.indptr)
    item_deg = np.array(R.getnnz(axis=0)).ravel()

    # Histogrammes log-log (attention aux zéros)
    for deg, title in [(user_deg, "Users"), (item_deg, "Items")]:
        deg = deg[deg > 0]
        vals, counts = np.unique(deg, return_counts=True)

        plt.figure(figsize=(6,4))
        plt.loglog(vals, counts, marker='o', linestyle='None')
        plt.title(f"Degree distribution (log-log) - {title}")
        plt.xlabel("degree")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(FIGURES2 / f"log_log_{Path(file).stem}.png", dpi=300)
        plt.close()


def top_item_centrality_subgraph(G_sub):
    users = [n for n,d in G_sub.nodes(data=True) if d.get("bipartite")==0]
    items = [n for n,d in G_sub.nodes(data=True) if d.get("bipartite")==1]
    U = len(users)

    item_deg = {i: G_sub.degree(i) for i in items}
    item_cent = {i: (item_deg[i] / U if U > 0 else 0.0) for i in items}

    top20 = sorted(item_cent.items(), key=lambda x: x[1], reverse=True)[:20]
    return top20, item_deg


def bipartite_clustering_subgraph(G_sub):
    # clustering biparti (Latapy et al.) : défini pour bipartite
    c = bipartite.clustering(G_sub)
    avg_c = sum(c.values()) / len(c) if len(c) else 0.0
    return avg_c, c


def connected_components_from_R(R):
    nU, nI = R.shape
    Zuu = sp.csr_matrix((nU, nU))
    Zii = sp.csr_matrix((nI, nI))

    A = sp.bmat([[Zuu, R],
                 [R.T, Zii]], format="csr")

    n_comp, labels = connected_components(A, directed=False, return_labels=True)

    # tailles des composantes
    sizes = np.bincount(labels)
    max_size = sizes.max() if sizes.size else 0
    return n_comp, int(max_size)


if __name__ == "__main__":

    pairs = [
        (csv, MATRIX / f"mat_csr_{csv.stem}.npz")
        for csv in INPUT.glob("*.csv")
        if (MATRIX / f"mat_csr_{csv.stem}.npz").exists()
    ]    

    for csv, npz in pairs:
        print("\n", csv.name, "<->", npz.name, "\n")
        bipartite_graph_gen(csv.name, npz.name)