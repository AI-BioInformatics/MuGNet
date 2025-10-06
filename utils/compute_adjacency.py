import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

def compute_representative_point(points, method="knn"):
    if "centroid" in method:
        return np.mean(points, axis=0)
    elif "medoid" in method:
        distances = cdist(points, points, metric='euclidean')
        return points[np.argmin(np.sum(distances, axis=1))]
    elif "knn" in method:  # Sostituito con il punto a massima densità con kNN
        n_neighbors = min(5, len(points))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
        distances, _ = nbrs.kneighbors(points)
        density_scores = np.mean(distances, axis=1)  # Media delle distanze ai k vicini
        return points[np.argmin(density_scores)]  # Punto con densità più alta
    else:
        raise ValueError("Metodo non valido. Scegli tra 'centroid', 'medoid', o 'kde'.")

def create_adjacency_matrix(
    data_pca,
    tissue_labels,
    gamma=0.5,
    method="knn",
    output_dir="."
):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Calcolo dei punti rappresentativi
    tissue_to_points = {}
    for point, tissue in zip(data_pca, tissue_labels):
        tissue_to_points.setdefault(tissue, []).append(point)

    tissue_representatives = {
        tissue: compute_representative_point(np.array(points), method)
        for tissue, points in tissue_to_points.items()
    }

    # 2. Matrice delle distanze tra rappresentativi
    tissue_names = list(tissue_representatives.keys())
    representative_matrix = np.array([tissue_representatives[t] for t in tissue_names])
    distance_matrix = cdist(representative_matrix, representative_matrix, metric='euclidean')

    # 3. Conversione in matrice di adiacenza con kernel gaussiano
    adjacency_matrix = np.exp(-gamma * distance_matrix**2)
    df_adjacency = pd.DataFrame(adjacency_matrix, index=tissue_names, columns=tissue_names)
    df_adjacency = df_adjacency.sort_index(axis=0).sort_index(axis=1)

    # 4. Salvataggio
    adjacency_matrix_path = os.path.join(output_dir, f"tissue_adjacency_matrix_{method}.csv")
    df_adjacency.to_csv(adjacency_matrix_path)
    
    
    
    






def create_patient_adjacency_matrices(patient_id, candidate_tissues, csv_file_path, threshold=0.1):
    """
    Crea una matrice di adiacenza per il paziente basata sui tessuti effettivamente presenti.
    Mantiene la corrispondenza tra node_id e tissue_id.

    :param patient_id: ID del paziente
    :param candidate_tissues: lista dei tessuti presenti nel paziente (es. ['Adn', 'Per', 'Bow', ...])
    :param csv_file_path: percorso del file CSV con la matrice di adiacenza globale
    :param threshold: valore minimo per considerare una connessione valida
    :return: matrice di adiacenza specifica per il paziente
    """
    # Carica la matrice globale di adiacenza
    matrice_globale = pd.read_csv(csv_file_path, index_col=0)

    # Estrarre solo i tessuti effettivamente presenti nel paziente
    adjacency_matrix = np.zeros((len(candidate_tissues), len(candidate_tissues)), dtype=float)

    for i, tissue in enumerate(candidate_tissues):
        for j, neighbor in enumerate(candidate_tissues):
            if tissue == neighbor:
                adjacency_matrix[i, j] = 1.0  # Diagonale principale (self-connection)
            else:
                # Se il tessuto è presente nel CSV, assegna il peso corrispondente
                if tissue in matrice_globale.index and neighbor in matrice_globale.columns:
                    weight = matrice_globale.loc[tissue, neighbor]
                    adjacency_matrix[i, j] = weight if weight >= threshold else 0.0
                    adjacency_matrix[j, i] = adjacency_matrix[i, j]  # Simmetria

    return adjacency_matrix