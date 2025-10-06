import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Per la riproducibilità
import torch
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
import pandas as pd
from collections import Counter
import random

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def ensure_directory_exists(directory):
    """ Crea la directory se non esiste """
    if not os.path.exists(directory):
        os.makedirs(directory)
        

def load_graphs_from_folder(folder_path):
    graphs = []
    patient_ids = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pt'):
            # Estrai ID paziente dalla parte prima del primo underscore
            patient_id = filename.split('_')[0]
            graph = torch.load(os.path.join(folder_path, filename))
            graph.patient_id = patient_id  # Aggiungi il patient id all'oggetto grafo
            graphs.append(graph)
            patient_ids.append(patient_id)
    return graphs, patient_ids


def aggregate_attention_scores(data, attn_weights, attn_weights1):
    """
    Aggrega gli attention scores per ogni nodo sorgente e destinazione 
    e li salva in quattro array distinti per il primo e secondo layer GAT.
    """
    attn_scores_np = attn_weights[1].cpu().detach().numpy()  # Estraggo gli scores del layer 1
    attn_scores_np1 = attn_weights1[1].cpu().detach().numpy()  # Estraggo gli scores del layer 2
    
    # Media sulle 4 head
    attn_scores_np = attn_scores_np.mean(axis=1)  # (num_edges,)
    attn_scores_np1 = attn_scores_np1.mean(axis=1)  # (num_edges,)

    num_nodes = data.x.shape[0]  # Numero totale di nodi

    # Inizializza i vettori per salvare i punteggi di attenzione per ogni nodo
    attn_scores_np_src = np.zeros(num_nodes)
    attn_scores_np_dst = np.zeros(num_nodes)
    attn_scores_np1_src = np.zeros(num_nodes)
    attn_scores_np1_dst = np.zeros(num_nodes)

    # Contatori per normalizzare i punteggi
    count_src = np.zeros(num_nodes)
    count_dst = np.zeros(num_nodes)

    # Itera sugli archi per sommare i punteggi e contare le occorrenze
    for i, (src, dst) in enumerate(data.edge_index.cpu().numpy().T):
        attn_scores_np_src[src] += attn_scores_np[i]
        attn_scores_np_dst[dst] += attn_scores_np[i]
        attn_scores_np1_src[src] += attn_scores_np1[i]
        attn_scores_np1_dst[dst] += attn_scores_np1[i]

        count_src[src] += 1
        count_dst[dst] += 1

    # Normalizza dividendo per il numero di connessioni
    attn_scores_np_src[count_src > 0] /= count_src[count_src > 0]
    attn_scores_np_dst[count_dst > 0] /= count_dst[count_dst > 0]
    attn_scores_np1_src[count_src > 0] /= count_src[count_src > 0]
    attn_scores_np1_dst[count_dst > 0] /= count_dst[count_dst > 0]

    return attn_scores_np_src, attn_scores_np_dst, attn_scores_np1_src, attn_scores_np1_dst



def load_graphs_from_folder_by_set_type(folder):
    import torch
    from torch_geometric.data import Data
    graphs = []
    for file in os.listdir(folder):
        if file.endswith('.pt'):
            graph = torch.load(os.path.join(folder, file))
            graphs.append(graph)
    return graphs


def init_weights(m):
    """Funzione per inizializzare i pesi del modello in modo deterministico."""
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)  # Xavier è ancora buono qui
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu', a=0.01)  # specifica la leaky ReLU
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


import json

def load_model_config(label, task, num_bins=None, adj_method=None, config_path="20_GitHub/best_params.json"):
    with open(config_path, "r") as f:
        config = json.load(f)

    if task == "classification":
        return config[label][task][adj_method]

    elif task == "regression":
        return config[label][task][str(num_bins)][adj_method]

    else:
        raise ValueError("Invalid task")
