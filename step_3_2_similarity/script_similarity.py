import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# ===================================================
# GESTION DE L'INPUT/OUTPUT
# ===================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT = ROOT / "outputs"
FIGURES = OUTPUT / "figures"
SIMILARITY = OUTPUT / "similarity"

FIGURES.mkdir(parents=True, exist_ok=True)
SIMILARITY.mkdir(parents=True, exist_ok=True)


# ===================================================
# 3.2.1 - IMPLÉMENTATION DES MESURES DE SIMILARITÉ
# ===================================================

def build_user_item_matrix(df):
    """
    Construit la matrice utilisateur-item sparse (CSR) à partir du dataframe.
    Retourne la matrice R, et les mappings user/item.
    """
    df = df.copy()
    df["user_idx"] = df["user_id"].astype("category").cat.codes
    df["item_idx"] = df["parent_asin"].astype("category").cat.codes

    n_users = df["user_idx"].nunique()
    n_items = df["item_idx"].nunique()

    # Gestion doublons : moyenne des ratings
    df_ui = df.groupby(["user_idx", "item_idx"], as_index=False)["rating"].mean()

    R = csr_matrix(
        (df_ui["rating"].to_numpy(dtype=np.float32),
         (df_ui["user_idx"].to_numpy(), df_ui["item_idx"].to_numpy())),
        shape=(n_users, n_items),
        dtype=np.float32
    )

    user_mapping = df[["user_id", "user_idx"]].drop_duplicates().set_index("user_idx")["user_id"]
    item_mapping = df[["parent_asin", "item_idx"]].drop_duplicates().set_index("item_idx")["parent_asin"]

    print(f"Matrice R : {R.shape} | nnz : {R.nnz}")
    return R, user_mapping, item_mapping


# --------------------------------------------------
# 1. Similarité Cosinus
# --------------------------------------------------

def similarite_cosinus(R, batch_size=500, seuil=0.01):
    """
    Calcule la similarité cosinus entre utilisateurs via sklearn (optimisé sparse).
    Seules les similarités > seuil sont conservées (stockage sparse).
    Retourne une matrice dense partielle sous forme de dict {(u,v): sim}.
    """
    print("\nCalcul similarité cosinus...")
    t0 = time.time()

    n_users = R.shape[0]
    similarities = {}

    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        batch = R[start:end]

        # Similarité cosinus entre le batch et tous les utilisateurs
        sim_batch = cosine_similarity(batch, R)  # shape (batch_size, n_users)

        for i, global_i in enumerate(range(start, end)):
            for j in range(n_users):
                if j <= global_i:
                    continue
                val = sim_batch[i, j]
                if val > seuil:
                    similarities[(global_i, j)] = float(val)

        if start % 5000 == 0:
            print(f"  Batch {start}/{n_users}...")

    t1 = time.time()
    print(f"  Temps : {t1 - t0:.2f}s | Paires stockées : {len(similarities)}")
    return similarities


# --------------------------------------------------
# 2. Corrélation de Pearson
# --------------------------------------------------

def similarite_pearson(R, batch_size=500, seuil=0.01):
    """
    Calcule la corrélation de Pearson entre utilisateurs par batch.
    Centrage par la moyenne de chaque utilisateur sur les items co-évalués.
    """
    print("\nCalcul similarité Pearson...")
    t0 = time.time()

    n_users = R.shape[0]
    R_dense = R  # on travaille ligne par ligne en sparse
    similarities = {}

    # Précalcul des moyennes par utilisateur (sur items évalués uniquement)
    means = np.zeros(n_users)
    for u in range(n_users):
        row = R.getrow(u)
        data = row.data
        means[u] = data.mean() if len(data) > 0 else 0.0

    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)

        for u in range(start, end):
            row_u = R.getrow(u)
            indices_u = set(row_u.indices)

            for v in range(u + 1, n_users):
                row_v = R.getrow(v)
                indices_v = set(row_v.indices)

                # Items co-évalués
                common = list(indices_u & indices_v)
                if len(common) < 2:
                    continue

                # Récupération des ratings sur les items communs
                r_u = np.array(row_u[:, common].todense()).flatten()
                r_v = np.array(row_v[:, common].todense()).flatten()

                # Centrage par la moyenne sur items communs
                mu_u = r_u.mean()
                mu_v = r_v.mean()

                num = np.sum((r_u - mu_u) * (r_v - mu_v))
                den = np.sqrt(np.sum((r_u - mu_u) ** 2)) * np.sqrt(np.sum((r_v - mu_v) ** 2))

                if den == 0:
                    continue

                val = num / den
                if abs(val) > seuil:
                    similarities[(u, v)] = float(val)

        if start % 1000 == 0:
            print(f"  Batch {start}/{n_users}...")

    t1 = time.time()
    print(f"  Temps : {t1 - t0:.2f}s | Paires stockées : {len(similarities)}")
    return similarities


# --------------------------------------------------
# 3. Similarité de Jaccard
# --------------------------------------------------

def similarite_jaccard(R, batch_size=500, seuil=0.01):
    """
    Calcule la similarité de Jaccard entre utilisateurs.
    Jaccard = |Iu ∩ Iv| / |Iu ∪ Iv|
    """
    print("\nCalcul similarité Jaccard...")
    t0 = time.time()

    n_users = R.shape[0]
    similarities = {}

    # Précalcul des ensembles d'items par utilisateur
    item_sets = {}
    for u in range(n_users):
        row = R.getrow(u)
        item_sets[u] = set(row.indices)

    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)

        for u in range(start, end):
            for v in range(u + 1, n_users):
                inter = len(item_sets[u] & item_sets[v])
                if inter == 0:
                    continue
                union = len(item_sets[u] | item_sets[v])
                val = inter / union if union > 0 else 0.0
                if val > seuil:
                    similarities[(u, v)] = float(val)

        if start % 1000 == 0:
            print(f"  Batch {start}/{n_users}...")

    t1 = time.time()
    print(f"  Temps : {t1 - t0:.2f}s | Paires stockées : {len(similarities)}")
    return similarities


# ===================================================
# 3.2.2 - ANALYSE COMPARATIVE
# ===================================================

def selectionner_5_utilisateurs(df):
    """
    Sélectionne 5 utilisateurs avec des profils variés :
    - 1 très actif (> 100 reviews)
    - 2 moyennement actifs (30-50 reviews)
    - 2 peu actifs (10-20 reviews)
    """
    counts = df["user_id"].value_counts()

    tres_actifs    = counts[counts > 100].index.tolist()
    moyens         = counts[(counts >= 30) & (counts <= 50)].index.tolist()
    peu_actifs     = counts[(counts >= 10) & (counts <= 20)].index.tolist()

    np.random.seed(42)
    selection = []

    if len(tres_actifs) >= 1:
        selection.append(np.random.choice(tres_actifs, 1)[0])
    if len(moyens) >= 2:
        selection.extend(np.random.choice(moyens, 2, replace=False).tolist())
    if len(peu_actifs) >= 2:
        selection.extend(np.random.choice(peu_actifs, 2, replace=False).tolist())

    print(f"\n5 utilisateurs sélectionnés : {selection}")
    for u in selection:
        print(f"  {u} : {counts[u]} reviews")

    return selection


def top_k_voisins(user_id, similarities, user_mapping, k=10):
    """
    Retourne les k voisins les plus similaires à user_id.
    """
    # Récupère l'index numérique de l'utilisateur
    user_idx = user_mapping[user_mapping == user_id].index[0]

    voisins = {}
    for (u, v), sim in similarities.items():
        if u == user_idx:
            voisins[v] = sim
        elif v == user_idx:
            voisins[u] = sim

    top_k = sorted(voisins.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(user_mapping[idx], sim) for idx, sim in top_k]


def tableau_comparatif_voisins(users, sim_cos, sim_pear, sim_jac, user_mapping, file, k=10):
    """
    Crée un tableau comparant les voisins identifiés par chaque mesure.
    Calcule aussi le coefficient de Jaccard entre ensembles de voisins.
    """
    print("\n===== TABLEAU COMPARATIF DES VOISINS =====")
    rows = []

    for user in users:
        voisins_cos  = set([v for v, _ in top_k_voisins(user, sim_cos,  user_mapping, k)])
        voisins_pear = set([v for v, _ in top_k_voisins(user, sim_pear, user_mapping, k)])
        voisins_jac  = set([v for v, _ in top_k_voisins(user, sim_jac,  user_mapping, k)])

        # Coefficient de Jaccard entre ensembles de voisins
        def jaccard_sets(a, b):
            return len(a & b) / len(a | b) if len(a | b) > 0 else 0.0

        j_cos_pear = jaccard_sets(voisins_cos, voisins_pear)
        j_cos_jac  = jaccard_sets(voisins_cos, voisins_jac)
        j_pear_jac = jaccard_sets(voisins_pear, voisins_jac)

        rows.append({
            "user_id": user,
            "voisins_cosinus":  str(list(voisins_cos)[:3])  + "...",
            "voisins_pearson":  str(list(voisins_pear)[:3]) + "...",
            "voisins_jaccard":  str(list(voisins_jac)[:3])  + "...",
            "jaccard(cos,pear)": round(j_cos_pear, 3),
            "jaccard(cos,jac)":  round(j_cos_jac,  3),
            "jaccard(pear,jac)": round(j_pear_jac, 3),
        })

    df_result = pd.DataFrame(rows)
    print(df_result.to_string(index=False))
    df_result.to_csv(SIMILARITY / f"comparaison_voisins_{file}.csv", index=False)
    print(f"\n  Sauvegardé : similarity/comparaison_voisins_{file}.csv")
    return df_result


def distribution_similarites(sim_cos, sim_pear, sim_jac, file):
    """
    Visualise la distribution des similarités pour chaque mesure (histogramme).
    Calcule moyenne, médiane, écart-type.
    """
    print("\n===== DISTRIBUTION DES SIMILARITÉS =====")

    mesures = {
        "Cosinus":  list(sim_cos.values()),
        "Pearson":  list(sim_pear.values()),
        "Jaccard":  list(sim_jac.values()),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (nom, valeurs) in zip(axes, mesures.items()):
        valeurs = np.array(valeurs)
        moyenne = valeurs.mean()
        mediane = np.median(valeurs)
        std     = valeurs.std()

        ax.hist(valeurs, bins=50, color="#3F51B5", alpha=0.75, edgecolor="white")
        ax.axvline(moyenne, color="red",    linestyle="--", label=f"Moyenne : {moyenne:.3f}")
        ax.axvline(mediane, color="orange", linestyle="--", label=f"Médiane : {mediane:.3f}")
        ax.set_title(f"Similarité {nom}", fontsize=13)
        ax.set_xlabel("Valeur de similarité")
        ax.set_ylabel("Fréquence")
        ax.legend(fontsize=9)
        ax.text(0.97, 0.95, f"σ = {std:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="gray")

        print(f"  {nom} — moyenne: {moyenne:.4f} | médiane: {mediane:.4f} | std: {std:.4f}")

    plt.suptitle(f"Distribution des similarités ({file})", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES / f"distribution_similarites_{file}.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegardé : figures/distribution_similarites_{file}.png")


def heatmap_similarites(R, sim_cos, user_mapping, file, n=30):
    """
    Crée une heatmap des similarités cosinus pour 30 utilisateurs aléatoires.
    """
    print(f"\n===== HEATMAP ({n} utilisateurs) =====")

    np.random.seed(42)
    n_users = R.shape[0]
    indices = np.random.choice(n_users, min(n, n_users), replace=False)

    # Construction de la matrice de similarité pour ces n utilisateurs
    sim_matrix = np.zeros((len(indices), len(indices)))

    for i, u in enumerate(indices):
        for j, v in enumerate(indices):
            if u == v:
                sim_matrix[i, j] = 1.0
            else:
                key = (min(u, v), max(u, v))
                sim_matrix[i, j] = sim_cos.get(key, 0.0)

    labels = [str(user_mapping.get(idx, idx))[:8] for idx in indices]

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        sim_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        vmin=0, vmax=1,
        linewidths=0.3,
        linecolor="lightgray"
    )
    plt.title(f"Heatmap similarité cosinus — {n} utilisateurs aléatoires\n({file})", fontsize=13)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0,  fontsize=6)
    plt.tight_layout()
    plt.savefig(FIGURES / f"heatmap_similarites_{file}.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegardé : figures/heatmap_similarites_{file}.png")


# ===================================================
# FONCTION PRINCIPALE
# ===================================================

def create_similarity(df, file):
    """
    Pipeline complet de la tâche 3.1 pour un fichier donné.
    """
    print(f"\n{'='*50}")
    print(f"TÂCHE 3.1 — MESURES DE SIMILARITÉ : {file}")
    print(f"{'='*50}")

    # Construction de la matrice
    R, user_mapping, item_mapping = build_user_item_matrix(df)

    # --- 3.2.1 : Calcul des trois mesures ---
    sim_cos  = similarite_cosinus(R)
    sim_pear = similarite_pearson(R)
    sim_jac  = similarite_jaccard(R)

    # --- 3.2.2 : Analyse comparative ---

    # Sélection des 5 utilisateurs
    users = selectionner_5_utilisateurs(df)

    # Tableau comparatif des voisins
    tableau_comparatif_voisins(users, sim_cos, sim_pear, sim_jac, user_mapping, file)

    # Distribution des similarités
    distribution_similarites(sim_cos, sim_pear, sim_jac, file)

    # Heatmap
    heatmap_similarites(R, sim_cos, user_mapping, file)

    print(f"\n✓ Tâche 3.1 terminée pour : {file}")
    return sim_cos, sim_pear, sim_jac


# ===================================================
# EXÉCUTION DIRECTE
# ===================================================

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    INPUT = ROOT / "input"

    file1 = "amazon_books_sample_active_users.csv"
    file2 = "amazon_books_sample_temporal.csv"

    print("Lecture du fichier 1...")
    df1 = pd.read_csv(INPUT / file1)
    create_similarity(df1, file1)

    print("Lecture du fichier 2...")
    df2 = pd.read_csv(INPUT / file2)
    create_similarity(df2, file2)