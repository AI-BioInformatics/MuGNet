import os
import torch
import pandas as pd
from torch_geometric.data import Data as geomData
from utils.compute_adjacency import create_patient_adjacency_matrices
from utils.functions import ensure_directory_exists


def generate_patient_graphs_from_split_regr(
    embeddings_path: str,
    unique_groups: list,
    split_csv_path: str,
    output_base_path: str,
    label_dict: dict,
    tissue_mapping_path: str,
    adjacency_base_path: str,
    adj_method: str
):
    """
    Crea i grafi paziente-specifici per una task di regressione con censura (es. C-Index).
    """
    # === Mappature tessuti
    tissue_name_to_idx = {t: i for i, t in enumerate(unique_groups)}
    idx_to_tissue_name = {i: t for t, i in tissue_name_to_idx.items()}

    # === Split e fold
    split_df = pd.read_csv(split_csv_path)
    split_info = {
        row["patient_id"]: (row["set"], row["fold"])
        for _, row in split_df.iterrows()
    }

    patient_graphs = {}

    # === Carica embeddings
    for fname in sorted(os.listdir(embeddings_path)):
        patient_id = fname.split('_')[0]
        tissue_code = fname.split('_')[1][1:4]

        if 'LN' in tissue_code:
            tissue_code = 'LN'
        if tissue_code == 'And':
            tissue_code = 'Adn'

        if tissue_code not in unique_groups or patient_id not in split_info:
            continue

        patient_graphs.setdefault(patient_id, []).append(
            (tissue_code, torch.load(os.path.join(embeddings_path, fname)), fname)
        )

    graphs = {}

    # === Crea grafi
    for patient_id, tissues in patient_graphs.items():
        if patient_id not in label_dict or patient_id not in split_info:
            continue

        set_type, fold = split_info[patient_id]
        adjacency_path = os.path.join(
            adjacency_base_path, set_type,
            f"fold_{fold}", f"tissue_adjacency_matrix_{adj_method}.csv"
        )
        adjacency_matrix = create_patient_adjacency_matrices(patient_id, unique_groups, adjacency_path)

        nodes, tissue_indices, edges, edge_weights = [], [], [], []

        # === Labels per regressione
        label_value_raw, c_value_raw = label_dict[patient_id]
        label_value = torch.tensor([label_value_raw], dtype=torch.long)
        c_value = torch.tensor([c_value_raw], dtype=torch.float)

        for tissue_code, embedding, _ in tissues:
            if embedding.ndim == 2 and embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)
            nodes.append(embedding)
            tissue_indices.append(tissue_name_to_idx[tissue_code])
        
        if adj_method == "knn_00":
            threshold = 0.1
            use_greater_equal = True
        else:
            threshold = 0.5
            use_greater_equal = False

        added_edges = set()
        for i, (t1, _, f1) in enumerate(tissues):
            for j, (t2, _, f2) in enumerate(tissues):
                if f1 == f2:
                    continue
                weight = adjacency_matrix[tissue_name_to_idx[t1], tissue_name_to_idx[t2]]
                if t1 == t2:
                    weight = 1.0 if (weight >= threshold if use_greater_equal else weight > threshold) else weight
                if (weight >= threshold if use_greater_equal else weight > threshold):
                    edge = (i, j) if i < j else (j, i)
                    if edge not in added_edges:
                        edges.append(edge)
                        edge_weights.append(weight)
                        added_edges.add(edge)

        if not nodes:
            continue

        x_tensor = torch.stack(nodes).to(torch.float32)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        tissue_tensor = torch.tensor(tissue_indices, dtype=torch.long)

        graph_data = geomData(
            x=x_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=label_value,
            c=c_value,
            tissue_idx=tissue_tensor
        )
        graph_data.patient_id = patient_id
        graphs[patient_id] = graph_data

        # === Salvataggio
        out_dir = os.path.join(output_base_path, f"fold_{fold}", set_type)
        ensure_directory_exists(out_dir)
        torch.save(graph_data, os.path.join(out_dir, f"{patient_id}_graph.pt"))
        print(f"✅ Grafo salvato: {patient_id} → fold {fold} ({set_type})")

    # === Salva mappatura tessuto ↔ indice
    torch.save(idx_to_tissue_name, os.path.join(tissue_mapping_path, "tissue_mapping.pt"))
    print("📁 Mappatura dei tessuti salvata in:", tissue_mapping_path)

    return graphs






def generate_patient_graphs_from_split_bin(
    embeddings_path: str,
    unique_groups: list,
    split_csv_path: str,
    output_base_path: str,
    label_dict: dict,
    tissue_mapping_path: str,
    adjacency_base_path: str,
    adj_method: str 
):
    """
    Crea grafi paziente-specifici per classificazione binaria (es. OS binario) a partire da uno split.
    Le matrici di adiacenza devono essere già precomputate per fold e set_type.
    """

    # === Mappature tessuti
    tissue_name_to_idx = {t: i for i, t in enumerate(unique_groups)}
    idx_to_tissue_name = {i: t for t, i in tissue_name_to_idx.items()}

    # === Split CSV
    split_df = pd.read_csv(split_csv_path)
    split_info = {
        row["patient_id"]: (row["set"], row["fold"])
        for _, row in split_df.iterrows()
    }

    patient_graphs = {}

    # === Carica embeddings
    for fname in sorted(os.listdir(embeddings_path)):
        patient_id = fname.split('_')[0]
        tissue_code = fname.split('_')[1][1:4]

        if 'LN' in tissue_code:
            tissue_code = 'LN'
        if tissue_code == 'And':
            tissue_code = 'Adn'

        if tissue_code not in unique_groups or patient_id not in split_info:
            continue

        patient_graphs.setdefault(patient_id, []).append(
            (tissue_code, torch.load(os.path.join(embeddings_path, fname)), fname)
        )

    graphs = {}

    # === Crea grafi
    for patient_id, tissues in patient_graphs.items():
        if patient_id not in label_dict or patient_id not in split_info:
            continue

        set_type, fold = split_info[patient_id]
        adjacency_path = os.path.join(
            adjacency_base_path, set_type,
            f"fold_{fold}", f"tissue_adjacency_matrix_{adj_method}.csv"
        )
        adjacency_matrix = create_patient_adjacency_matrices(patient_id, unique_groups, adjacency_path)

        nodes, tissue_indices, edges, edge_weights = [], [], [], []

        label_value = label_dict[patient_id]
        label_value = torch.tensor([label_value], dtype=torch.long)

        for tissue_code, embedding, _ in tissues:
            if embedding.ndim == 2 and embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)
            nodes.append(embedding)
            tissue_indices.append(tissue_name_to_idx[tissue_code])

        if adj_method == "knn_00":
            threshold = 0.1
            use_greater_equal = True
        else:
            threshold = 0.5
            use_greater_equal = False
        
        added_edges = set()
        for i, (t1, _, f1) in enumerate(tissues):
            for j, (t2, _, f2) in enumerate(tissues):
                if f1 == f2:
                    continue
                weight = adjacency_matrix[tissue_name_to_idx[t1], tissue_name_to_idx[t2]]
                if t1 == t2:
                    weight = 1.0 if (weight >= threshold if use_greater_equal else weight > threshold) else weight
                if (weight >= threshold if use_greater_equal else weight > threshold):
                    edge = (i, j) if i < j else (j, i)
                    if edge not in added_edges:
                        edges.append(edge)
                        edge_weights.append(weight)
                        added_edges.add(edge)

        if not nodes:
            continue

        x_tensor = torch.stack(nodes).to(torch.float32)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        tissue_tensor = torch.tensor(tissue_indices, dtype=torch.long)

        graph_data = geomData(
            x=x_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=label_value,
            tissue_idx=tissue_tensor
        )
        graph_data.patient_id = patient_id
        graphs[patient_id] = graph_data

        # === Salva il grafo
        out_dir = os.path.join(output_base_path, f"fold_{fold}", set_type)
        ensure_directory_exists(out_dir)
        torch.save(graph_data, os.path.join(out_dir, f"{patient_id}_graph.pt"))
        print(f"✅ Grafo salvato: {patient_id} → fold {fold} ({set_type})")

    # === Salva mappatura tessuto ↔ indice
    torch.save(idx_to_tissue_name, os.path.join(tissue_mapping_path, "tissue_mapping.pt"))
    print("📁 Mappatura dei tessuti salvata in:", tissue_mapping_path)

    return graphs
