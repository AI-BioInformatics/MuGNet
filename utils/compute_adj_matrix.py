import os
import numpy as np
import pandas as pd
from .load_embedding_files import load_embeddings_nact, load_embeddings_pds
from .compute_adjacency import create_adjacency_matrix

from sklearn.decomposition import PCA

def apply_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data)
    return transformed, pca


def compute_adjacency_from_split_knn(
    embeddings_path: str,
    split_csv_path: str,
    output_base_dir: str,
    allowed_tissues: list,
    gamma: float = 0.5,
    method: str = "knn"
):
    print(f"🔍 Caricamento embeddings da: {embeddings_path}")
    data, patient_labels, tissue_labels = load_embeddings_nact(embeddings_path)
    data = data.reshape(-1, 768)
    df_splits = pd.read_csv(split_csv_path)

    for fold in sorted(df_splits["fold"].unique()):
        for set_type in ["train", "val", "test"]:
            print(f"\n📦 Fold {fold} | Set: {set_type.upper()}")
            patients = df_splits[
                (df_splits['fold'] == fold) & (df_splits['set'] == set_type)
            ]['patient_id'].tolist()

            mask = np.isin(patient_labels, patients)
            data_fold = data[mask]
            tissue_labels_fold = np.array(tissue_labels)[mask]

            if len(data_fold) == 0:
                print(f"⚠️ Nessun embedding trovato per fold {fold} ({set_type}), skip.")
                continue

            # Filtro sui tessuti validi
            tissue_mask = np.isin(tissue_labels_fold, allowed_tissues)
            data_fold = data_fold[tissue_mask]
            tissue_labels_fold = tissue_labels_fold[tissue_mask]

            print("⚙️ PCA in corso...")
            data_pca, pca_model = apply_pca(data_fold, n_components=2)
            print(f"Varianza spiegata: {np.cumsum(pca_model.explained_variance_ratio_)}")

            output_dir = os.path.join(output_base_dir, set_type, f"fold_{fold}")
            os.makedirs(output_dir, exist_ok=True)

            create_adjacency_matrix(
                data_pca,
                tissue_labels_fold,
                gamma=gamma,
                method=method,
                output_dir=output_dir
            )



def compute_adjacency_from_split_corr(
    embeddings_path: str,
    split_csv_path: str,
    output_base_dir: str,
    allowed_tissues: list
):
    from sklearn.decomposition import PCA
    from .load_embedding_files import load_embeddings_nact
    import numpy as np
    import pandas as pd
    import os

    print(f"🔍 Caricamento embeddings da: {embeddings_path}")
    data, patient_labels, tissue_labels = load_embeddings_nact(embeddings_path)
    data = data.reshape(-1, 768)
    df_splits = pd.read_csv(split_csv_path)

    folds = df_splits["fold"].unique()

    for fold in sorted(folds):
        for set_type in ['train', 'val', 'test']:
            print(f"\n📦 Fold {fold} - Processing set: {set_type.upper()}")
            patients = df_splits[
                (df_splits["fold"] == fold) & (df_splits["set"] == set_type)
            ]["patient_id"].tolist()

            mask = np.isin(patient_labels, patients)
            data_fold = data[mask]
            tissue_labels_fold = np.array(tissue_labels)[mask]

            # Filtra per tessuti ammessi
            tissue_mask = np.isin(tissue_labels_fold, allowed_tissues)
            data_fold = data_fold[tissue_mask]
            tissue_labels_fold = tissue_labels_fold[tissue_mask]

            if len(data_fold) == 0:
                print(f"⚠️ Nessun embedding trovato per fold {fold} set {set_type}, skip.")
                continue

            print("⚙️ PCA in corso...")
            pca = PCA(n_components=2)
            data_pca = pca.fit_transform(data_fold)
            print(f"📊 Varianza spiegata cumulativa: {np.cumsum(pca.explained_variance_ratio_)}")

            # === Calcolo della matrice di correlazione binaria tra i tessuti
            df = pd.DataFrame(data_pca, columns=["PC1", "PC2"])
            df["Tissue"] = tissue_labels_fold

            pc_mean = df.groupby("Tissue").mean()
            correlation_matrix = pc_mean.T.corr()

            # Binarizzazione della correlazione
            adjacency_matrix = np.where(correlation_matrix == -1.0, 0.0, 1.0)
            adjacency_df = pd.DataFrame(adjacency_matrix, index=correlation_matrix.index, columns=correlation_matrix.columns)

            output_dir = os.path.join(output_base_dir, set_type, f"fold_{fold}")
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"tissue_adjacency_matrix_pca_corr.csv")
            adjacency_df.to_csv(out_path)

            print(f"✅ Matrice salvata in: {out_path}")
