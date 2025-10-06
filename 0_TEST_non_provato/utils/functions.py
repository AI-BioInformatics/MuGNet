from lifelines.utils import concordance_index
import json
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Per la riproducibilità
import torch
import numpy as np
import random

def compute_c_index(y, pred, c):
    return concordance_index(y, -pred, c)


def load_model_config(label, task, num_bins=None, adj_method=None, config_path="20_GitHub/TEST/best_params.json"):
    with open(config_path, "r") as f:
        config = json.load(f)

    if task == "classification":
        return config[label][task][adj_method]
    
    elif task == "regression":
        return config[label][task][str(num_bins)][adj_method]
    
    else:
        raise ValueError("Invalid task")

    
    
def load_graphs_from_folder_by_set_type(folder):
    import torch
    from torch_geometric.data import Data
    graphs = []
    for file in os.listdir(folder):
        if file.endswith('.pt'):
            graph = torch.load(os.path.join(folder, file))
            graphs.append(graph)
    return graphs

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