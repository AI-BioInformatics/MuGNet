import argparse
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from utils.labels import os_discrete_balanced, os_binary, hr_binary, pfi_discrete
from utils.compute_adj_matrix import compute_adjacency_from_split_knn, compute_adjacency_from_split_corr
from utils.create_graphs import generate_patient_graphs_from_split_regr, generate_patient_graphs_from_split_bin
from train_binary import run_binary_training
from train_regr import run_regr_training
from utils.functions import set_seed, load_model_config

def ensure_directory_exists(path):
    os.makedirs(path, exist_ok=True)

def create_stratified_folds_from_label_dict(label_dict, n_splits=5, random_state=43):
    patient_ids = list(label_dict.keys())
    label_bins = [label_dict[pid] if not isinstance(label_dict[pid], tuple) else label_dict[pid][0] for pid in patient_ids]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_folds = []

    for fold_idx, (train_val_idx, val_idx) in enumerate(skf.split(patient_ids, label_bins)):
        train_val_ids = [patient_ids[i] for i in train_val_idx]
        val_ids = [patient_ids[i] for i in val_idx]
        train_val_labels = [label_bins[i] for i in train_val_idx]

        train_ids, test_ids = train_test_split(
            train_val_ids,
            test_size=0.125,
            stratify=train_val_labels,
            random_state=random_state
        )

        fold_data = []
        for pid in train_ids:
            fold_data.append({'patient_id': pid, 'set': 'train', 'fold': fold_idx + 1})
        for pid in val_ids:
            fold_data.append({'patient_id': pid, 'set': 'val', 'fold': fold_idx + 1})
        for pid in test_ids:
            fold_data.append({'patient_id': pid, 'set': 'test', 'fold': fold_idx + 1})

        all_folds.extend(fold_data)
        print(f"✔️ Fold {fold_idx + 1}: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

    return pd.DataFrame(all_folds)

def create_labels_df(os_dict):
    records = []
    for pid, label in os_dict.items():
        if isinstance(label, tuple):
            class_label, c = label
            records.append({'patient_id': pid, 'label': class_label, 'c': c})
        else:
            records.append({'patient_id': pid, 'label': label})
    return pd.DataFrame(records)

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

    return parser.parse_args()

def main():
    args = parse_args()
    
    # === Costruzione dinamica dell'output_dir ===
    if args.task == "regression":
        output_dir_name = f"{args.label}_{args.task}_{args.adj_method}_{args.num_bins}bin"
    else:
        output_dir_name = f"{args.label}_{args.task}_{args.adj_method}"

    args.output_dir = os.path.join(args.output_dir, output_dir_name)

    labels_path = "docs/labels.xlsx"
    labels_path_hr = "docs/HR_dataset.csv"
    embeddings_path_nact = "data/slide_embeddings_TITAN"
    embedding_path_pds = "data/slide_embeddings_TITAN"
    tissue_list = ['Adn', 'Per', 'Ome', 'Tub', 'Ova', 'Ute', 'Vag', 'Bow', 'Mes', 'LN']

    # Output paths
    splits_path = os.path.join(args.output_dir, "splits/kfold_patient_splits.csv")
    labels_out_path = os.path.join(args.output_dir, "splits/patient_labels.csv")
    adjacency_path = os.path.join(args.output_dir, f"adjacency_matrices/{args.adj_method}")
    graphs_out_path = os.path.join(args.output_dir, "graphs")

    ensure_directory_exists(os.path.dirname(splits_path))
    ensure_directory_exists(adjacency_path)
    ensure_directory_exists(graphs_out_path)
    
    model_config = load_model_config(
        label=args.label,
        task=args.task,
        num_bins=args.num_bins,
        adj_method=args.adj_method
    )

    # === Labels
    if args.task == "classification":
        print(f"🔍 Using binary {args.label.upper()} classification labels")
        
        if args.label.upper() == "OS":
            embeddings_path = embeddings_path_nact
            label_dict = os_binary(labels_path, embeddings_path)
            
        elif args.label.upper() == "HR":
            embeddings_path = embedding_path_pds
            label_dict = hr_binary(labels_path_hr, embeddings_path)
            
        elif args.label.upper() == "PFI":
            raise ValueError("❌ PFI label available only for regression task.")
        
        df_splits = create_stratified_folds_from_label_dict(label_dict)
        df_labels = create_labels_df(label_dict)

        df_splits.to_csv(splits_path, index=False)
        df_labels.to_csv(labels_out_path, index=False)
        print(f"📁 Saved splits to {splits_path}")
        print(f"📁 Saved labels to {labels_out_path}")

        # === Adjacency matrices
        print(f"🔧 Generating adjacency matrices with method: {args.adj_method}")
        if "knn" in args.adj_method:
            compute_adjacency_from_split_knn(
                embeddings_path=embeddings_path,
                split_csv_path=splits_path,
                output_base_dir=adjacency_path,
                allowed_tissues=tissue_list,
                method=args.adj_method
            )
        elif args.adj_method == "pca_corr":
            compute_adjacency_from_split_corr(
                embeddings_path=embeddings_path,
                split_csv_path=splits_path,
                output_base_dir=adjacency_path,
                allowed_tissues=tissue_list
            )
            
        elif args.adj_method.lower() == "anat":
            import shutil

            source_file = "20_GitHub/anatomic_adjacency_matrix.csv"

            if not os.path.exists(source_file):
                raise FileNotFoundError(f"❌ Matrice anatomica non trovata in: {source_file}")

            for fold in range(1, 6): 
                for set_type in ["train", "val", "test"]:
                    target_dir = os.path.join(adjacency_path, set_type, f"fold_{fold}")
                    ensure_directory_exists(target_dir)

                    dest_file = os.path.join(target_dir, f"tissue_adjacency_matrix_{args.adj_method}.csv")

                    if not os.path.exists(dest_file):
                        shutil.copy(source_file, dest_file)
                        print(f"📦 Matrice anatomica copiata in: {dest_file}")
                    else:
                        print(f"✅ Matrice già presente in: {dest_file}")

        print("✅ Adjacency matrices created.")

        # === Graphs
        print("📈 Generating patient graphs...")
        folds = df_splits["fold"].unique()

        for fold in folds:
            set_seed(43)
            fold_csv_path = os.path.join(args.output_dir, f"splits/split_fold_{fold}.csv")
            df_splits[df_splits["fold"] == fold].to_csv(fold_csv_path, index=False)

            generate_patient_graphs_from_split_bin(
                embeddings_path=embeddings_path,
                unique_groups=tissue_list,
                split_csv_path=fold_csv_path,
                output_base_path=graphs_out_path,
                label_dict=label_dict,
                tissue_mapping_path=graphs_out_path,
                adjacency_base_path=adjacency_path,
                adj_method=args.adj_method
            )
        
        graph_dir = os.path.join(args.output_dir, "graphs")
        run_binary_training(
            folds,
            graphs_dir=graph_dir,
            save_dir=args.output_dir,
            output_size=1,
            embedding_size=model_config["embedding_size"],
            num_heads=model_config["num_heads"],
            learning_rate=model_config["learning_rate"],
            weight_decay=model_config["weight_decay"]
        )


        
    
    else:
        print(f"🔍 Using discretized {args.label.upper()} labels with {args.num_bins} bins")
        embeddings_path = embeddings_path_nact
        if args.label.upper() == "OS":
            label_dict = os_discrete_balanced(labels_path, num_bins=args.num_bins, embedding_dir=embeddings_path)
            
        elif args.label.upper() == "PFI":
            if args.num_bins != 3:
                raise ValueError("❌ PFI supporta solo 3 bin. Imposta --num_bins 3.")
            label_dict = pfi_discrete(labels_path, embeddings_path)
            
        elif args.label.upper() == "HR":
            raise ValueError("❌ La label HR è disponibile solo per il task di classificazione.")

        df_splits = create_stratified_folds_from_label_dict(label_dict)
        df_labels = create_labels_df(label_dict)

        df_splits.to_csv(splits_path, index=False)
        df_labels.to_csv(labels_out_path, index=False)
        print(f"📁 Saved splits to {splits_path}")
        print(f"📁 Saved labels to {labels_out_path}")

        # === Adjacency
        print(f"🔧 Generating adjacency matrices with method: {args.adj_method}")
        if "knn" in args.adj_method:
            compute_adjacency_from_split_knn(
                embeddings_path=embeddings_path,
                split_csv_path=splits_path,
                output_base_dir=adjacency_path,
                allowed_tissues=tissue_list,
                method=args.adj_method
            )
        elif args.adj_method == "pca_corr":
            compute_adjacency_from_split_corr(
                embeddings_path=embeddings_path,
                split_csv_path=splits_path,
                output_base_dir=adjacency_path,
                allowed_tissues=tissue_list
            )
            
        elif args.adj_method.lower() == "anat":
            import shutil

            source_file = "20_GitHub/anatomic_adjacency_matrix.csv"

            if not os.path.exists(source_file):
                raise FileNotFoundError(f"❌ Matrice anatomica non trovata in: {source_file}")
            folds = df_splits["fold"].unique()

            for fold in folds: 
                for set_type in ["train", "val", "test"]:
                    target_dir = os.path.join(adjacency_path, set_type, f"fold_{fold}")
                    ensure_directory_exists(target_dir)

                    dest_file = os.path.join(target_dir, f"tissue_adjacency_matrix_{args.adj_method}.csv")

                    if not os.path.exists(dest_file):
                        shutil.copy(source_file, dest_file)
                        print(f"📦 Matrice anatomica copiata in: {dest_file}")
                    else:
                        print(f"✅ Matrice già presente in: {dest_file}")

            
            

        print("✅ Adjacency matrices created.")

        # === Graphs
        print("📈 Generating patient graphs...")
        folds = df_splits["fold"].unique()

        for fold in folds:
            set_seed(43)
            fold_csv_path = os.path.join(args.output_dir, f"splits/split_fold_{fold}.csv")
            df_splits[df_splits["fold"] == fold].to_csv(fold_csv_path, index=False)
        
            generate_patient_graphs_from_split_regr(
                embeddings_path=embeddings_path,
                unique_groups=tissue_list,
                split_csv_path=fold_csv_path,
                output_base_path=graphs_out_path,
                label_dict=label_dict,
                tissue_mapping_path=graphs_out_path,
                adjacency_base_path=adjacency_path,
                adj_method=args.adj_method
            )

        print("✅ Patient graphs generated.")
        
        graph_dir = os.path.join(args.output_dir, "graphs")
        run_regr_training(
            folds,
            graphs_dir=graph_dir,
            save_dir=args.output_dir,
            output_size=args.num_bins,
            embedding_size=model_config["embedding_size"],
            num_heads=model_config["num_heads"],
            learning_rate=model_config["learning_rate"],
            weight_decay=model_config["weight_decay"]
        )


if __name__ == "__main__":
    main()
