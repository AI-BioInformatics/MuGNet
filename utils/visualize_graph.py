import torch
import networkx as nx
import matplotlib.pyplot as plt
import os
from torch_geometric.utils import to_networkx

def visualize_graph(graph_data, output_path="graph.png"):
    """
    Funzione per visualizzare e salvare un grafo PyG con miglioramenti nella disposizione dei nodi.
    """
    print("Edge Index:\n", graph_data.edge_index)
    print("Numero di nodi:", graph_data.x.shape[0])

    # Controlla se il grafo ha archi
    if graph_data.edge_index.numel() == 0:
        print("⚠️ Attenzione: Il grafo non ha archi!")
        return

    # Converte il grafo PyG in un grafo NetworkX
    nx_graph = to_networkx(graph_data, to_undirected=True)

    # Disegna il grafo
    plt.figure(figsize=(10, 8))

    # Usa un layout che distribuisce meglio i nodi
    pos = nx.kamada_kawai_layout(nx_graph)  # Disposizione Kamada-Kawai, distribuisce meglio i nodi
    # pos = nx.spring_layout(nx_graph, k=0.15)  # Alternativa con layout spring, regola il parametro k per distanza tra i nodi

    # Disegna i nodi e gli archi con opzioni personalizzate
    nx.draw(nx_graph, pos, with_labels=False, node_color='lightblue', edge_color='gray', node_size=1000, font_size=12, width=2)
    
    labels = {i: graph_data.tissue_idx[i].item() for i in range(graph_data.x.shape[0])}
    nx.draw_networkx_labels(nx_graph, pos, labels=labels, font_size=12)


    # Aggiungi i pesi agli archi se presenti
    if graph_data.edge_attr is not None:
        edge_labels = {(i, j): f"{w:.2f}" for (i, j), w in zip(graph_data.edge_index.t().tolist(), graph_data.edge_attr.tolist())}
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_size=10)

    # Salva l'immagine
    plt.savefig(output_path, format="png", dpi=300)
    print(f"Grafico salvato in {output_path}")
    
    
def visualize_graph_with_tissue(graph_data, output_path="graph.png", tissue_map=None):
    if graph_data.edge_index.numel() == 0:
        print("⚠️ Attenzione: Il grafo non ha archi!")
        return

    nx_graph = to_networkx(graph_data, to_undirected=True)
    pos = nx.kamada_kawai_layout(nx_graph)

    plt.figure(figsize=(10, 8))
    nx.draw(
        nx_graph,
        pos,
        with_labels=False,
        node_color='lightblue',
        edge_color='gray',
        node_size=1000,
        width=2
    )

    # 🔽 Etichette dei nodi usando tissue_map
    labels = {}
    for i in range(graph_data.x.shape[0]):
        tissue_idx = graph_data.tissue_idx[i].item()
        tissue_name = tissue_map.get(tissue_idx, f"ID {tissue_idx}")
        labels[i] = tissue_name

    nx.draw_networkx_labels(nx_graph, pos, labels=labels, font_size=10)

    # 🔽 Etichette sugli archi (pesi)
    if graph_data.edge_attr is not None:
        edge_labels = {
            (i, j): f"{w:.2f}"
            for (i, j), w in zip(graph_data.edge_index.t().tolist(), graph_data.edge_attr.tolist())
        }
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_size=10)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, format="png", dpi=300)
    plt.close()
    print(f"✅ Grafico salvato in {output_path}")



# Percorso della cartella con i file .pt
input_dir = "20_GitHub/outputs/anat_3/graphs"
output_dir = "20_GitHub/outputs/anat_3/graphs_imgs"

import pandas as pd
import torch

tissue_map = torch.load("20_GitHub/outputs/anat_3/graphs/tissue_mapping.pt")


# Crea la cartella di output se non esiste
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Walk ricorsivo
for root, dirs, files in os.walk(input_dir):
    for file_name in files:
        if file_name.endswith(".pt") and file_name != "tissue_mapping.pt":
            graph_path = os.path.join(root, file_name)
            graph_data = torch.load(graph_path)

            # Output path che replica la struttura
            rel_path = os.path.relpath(graph_path, input_dir)  # es: fold_1/test/D368_graph.pt
            output_path = os.path.join(output_dir, rel_path).replace(".pt", ".png")

            # Crea la cartella di output se non esiste
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            visualize_graph_with_tissue(graph_data, output_path=output_path, tissue_map=tissue_map)
