# test_pipeline.py
import os
import torch
from torch_geometric.loader import DataLoader
from model import GAT_with_GAP_norm_leakyRELU_06_GraphNorm
from utils.functions import load_graphs_from_folder_by_set_type, load_model_config, set_seed
from utils.test_utils import evaluate_model
import argparse
import numpy as np
import pandas as pd
from torch.nn import BCEWithLogitsLoss
from NLL_loss import NLLSurvLoss

def run_test_on_fold(adj_method, label, fold, base_dir, task, num_bins):
    # === Gestione dei path
    if task == "regression" and label == "os":
        bin_folder = f"{num_bins}bin"
        graphs_dir = os.path.join(base_dir, task, label, bin_folder, adj_method, f"fold_{fold}")
        model_path = os.path.join("pre_trained", task, label, bin_folder, adj_method, f"best_model_fold_{fold}.pth")
        pos_csv = os.path.join("pre_trained", task, label, bin_folder, adj_method, "pos_weights_per_fold.csv")
    else:
        graphs_dir = os.path.join(base_dir, task, label, adj_method, f"fold_{fold}")
        model_path = os.path.join("pre_trained", task, label, adj_method, f"best_model_fold_{fold}.pth")
        pos_csv = os.path.join("pre_trained", task, label, adj_method, "pos_weights_per_fold.csv")

    test_graphs = load_graphs_from_folder_by_set_type(os.path.join(graphs_dir, "test"))
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)


    input_dim = test_graphs[0].x.shape[1]
    #PER PFI E HR SOLO KNN 00 è reale gli altri ho messo 64 - 4
    model_config = load_model_config(label, task, num_bins, adj_method)

    
    model = GAT_with_GAP_norm_leakyRELU_06_GraphNorm(
        input_dim=input_dim,
        embedding_size=model_config["embedding_size"],
        num_heads=model_config["num_heads"],
        output_size=1 if task == "classification" else num_bins
    )
    
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    if task == "classification":
        # Load pos_weight from CSV
        pos_csv = os.path.join("./pre_trained", task, label, adj_method, "pos_weights_per_fold.csv")
        if os.path.exists(pos_csv):
            df_pos = pd.read_csv(pos_csv)
            pos_value = df_pos.loc[df_pos["fold"] == fold, "pos_weight"].values[0]
            pos_weight = torch.tensor([pos_value], dtype=torch.float32).to(device)
            print(f"📐 Using pos_weight={pos_value:.4f} for fold {fold}")
        else:
            pos_weight = torch.tensor([1.0], dtype=torch.float32).to(device)
            print("⚠️ pos_weight CSV not found, defaulting to 1.0")

        criterion = BCEWithLogitsLoss(pos_weight=pos_weight)

        
    elif task == "regression":
        criterion = NLLSurvLoss(alpha=0.0, reduction="mean")
        
    metrics = evaluate_model(model, test_loader, criterion, device, task)

    return metrics

def run_all_folds(adj_method, label, base_dir, output_dir, task, num_bins):
    all_metrics = []
    print(f"📊 Running test for all 5 folds | Label: {label} | Adjacency: {adj_method}")

    for fold in range(1, 6):
        set_seed(43)
        print(f"\n🔹 Fold {fold}")
        metrics = run_test_on_fold(adj_method, label, fold, base_dir, task, num_bins)
        metrics["fold"] = fold
        for k, v in metrics.items():
            if k != "fold":
                print(f"{k}: {v:.4f}")
        all_metrics.append(metrics)

    # === Calcola media e std
    print("\n📈 Summary across 5 folds:")
    summary = {}
    for metric in all_metrics[0].keys():
        if metric == "fold":
            continue
        values = [m[metric] for m in all_metrics]
        mean = np.mean(values)
        std = np.std(values)
        summary[metric] = {"mean": mean, "std": std}
        print(f"{metric}: {mean:.4f} ± {std:.4f}")

    # === Salvataggio CSV

    per_fold_path = os.path.join(output_dir, "results_per_fold.csv")
    summary_path = os.path.join(output_dir, "results_summary.csv")

    df_all = pd.DataFrame(all_metrics)
    df_all.to_csv(per_fold_path, index=False)

    df_summary = pd.DataFrame([
        {"metric": k, "mean": v["mean"], "std": v["std"]}
        for k, v in summary.items()
    ])
    df_summary.to_csv(summary_path, index=False)

    print(f"\n📁 Saved per-fold results to: {per_fold_path}")
    print(f"📁 Saved summary results to: {summary_path}")

    return summary


def parse_args():
        parser = argparse.ArgumentParser(description="Pipeline setup for MuGNet")
        parser.add_argument("--task", type=str, choices=["classification", "regression"], default="classification",
                            help="Prediction task type")
        parser.add_argument("--label", type=str, choices=["os", "pfi", "hr"], default="os",
                            help="Label type")
        parser.add_argument("--adj_method", type=str, choices=["knn_00", "knn_05", "anat", "pca_corr"], default="knn",
                            help="Adjacency matrix construction method")
        parser.add_argument("--num_bins", type=int, default=3,
                            help="Number of bins for survival regression (ignored for classification)")
        parser.add_argument("--output_dir", type=str, default="outputs/",
                            help="Base directory for saving outputs")
        parser.add_argument("--base_dir", default="graphs", help="Base path to graph data")
        
        return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    
    # === Costruzione dinamica dell'output_dir ===
    if args.task == "regression":
        if args.label == "os":
            output_dir_name = f"{args.label}_{args.task}_{args.adj_method}_{args.num_bins}bin"
        else:
            output_dir_name = f"{args.label}_{args.task}_{args.adj_method}"
    else:
        output_dir_name = f"{args.label}_{args.task}_{args.adj_method}"

    args.output_dir = os.path.join(args.output_dir, output_dir_name)
    
    
    
    if args.task == "classification":
        print(f"🔍 Using binary {args.label.upper()} classification labels")
        
        if args.label.upper() == "PFI":
            raise ValueError("❌ La label PFI è disponibile solo per il task di regressione.")

        run_all_folds(args.adj_method, args.label, args.base_dir, args.output_dir, args.task, args.num_bins)
        
        
    else:
        print(f"🔍 Using discretized {args.label.upper()} labels with {args.num_bins} bins")
        
        if args.label.upper() == "PFI":
            if args.num_bins != 3:
                raise ValueError("❌ PFI supporta solo 3 bin. Imposta --num_bins 3.")
            
        elif args.label.upper() == "HR":
            raise ValueError("❌ La label HR è disponibile solo per il task di classificazione.")
        
        run_all_folds(args.adj_method, args.label, args.base_dir, args.output_dir, args.task, args.num_bins)
