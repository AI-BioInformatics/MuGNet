import os
import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
from model import GAT_with_GAP_norm_leakyRELU_06_GraphNorm
from utils.functions import set_seed, ensure_directory_exists, load_graphs_from_folder_by_set_type, init_weights
from utils.loss_visualization import loss_show, plot_metrics, plot_roc_curve, plot_confusion_matrix
from utils.train_eval_test_binary import train, test  # Assicurati che queste siano importabili
import pandas as pd

def run_binary_training(folds, graphs_dir, save_dir, output_size,
                        embedding_size, num_heads, learning_rate, weight_decay):

    pos_weights_data = []

    for fold in folds:
        set_seed(43)
        print(f"\n🔹 Fold {fold} in corso... [{fold}/{folds}]\n")

        # === Carica i grafi
        def load_graphs(set_type):
            path = os.path.join(graphs_dir, f"fold_{fold}", set_type)
            return load_graphs_from_folder_by_set_type(path)

        train_graphs = load_graphs("train")
        val_graphs = load_graphs("val")
        test_graphs = load_graphs("test")

        print(f"Train: {len(train_graphs)} | Val: {len(val_graphs)} | Test: {len(test_graphs)}")

        # === DataLoader
        train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

        # === Model
        input_dim = train_graphs[0].x.shape[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GAT_with_GAP_norm_leakyRELU_06_GraphNorm(
                    input_dim=input_dim,
                    embedding_size=embedding_size,
                    num_heads=num_heads,
                    output_size=output_size
                ).to(device)
        model.apply(init_weights)

        # === Loss e optimizer
        train_labels = torch.cat([g.y for g in train_graphs])
        class_counts = torch.bincount(train_labels.long())
        pos_weight_value = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1.0
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
        pos_weights_data.append({"fold": fold, "pos_weight": pos_weight_value})

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # === Training
        train_losses, val_losses = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            fold=fold,
            optimizer=optimizer,
            device=device,
            criterion=criterion,
            save_dir=save_dir,
            epochs=50,
            patience=10,
            lr_patience=5
        )
        

        # === Save model
        model_dir = os.path.join(save_dir, "best_params", "model_weights")
        ensure_directory_exists(model_dir)
        torch.save(model.state_dict(), os.path.join(model_dir, f"model_fold_{fold}.pth"))

        # === Plot loss
        img_dir = os.path.join(save_dir, "best_params", "imgs", f"fold_{fold}", "test")
        ensure_directory_exists(img_dir)
        loss_show(train_losses, val_losses, img_dir)

        # === Evaluation
        model.load_state_dict(torch.load(os.path.join(model_dir, f"model_fold_{fold}.pth")))
        _, acc, prec, rec, f1, auc, cm, all_labels, all_probs, all_ids, all_preds = test(
            model, test_loader, device, criterion
        )

        plot_metrics([acc, prec, rec, f1, auc], ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'], img_dir)
        plot_roc_curve(all_labels, all_probs, img_dir)
        plot_confusion_matrix(cm, img_dir)
        

    # === Salva pesi positivi
    pos_dir = os.path.join(save_dir, "best_params", "pos_weight")
    ensure_directory_exists(pos_dir)
    pd.DataFrame(pos_weights_data).to_csv(os.path.join(pos_dir, "pos_weights_per_fold.csv"), index=False)
    print("✅ File dei pos_weight salvato.")

    # === Unifica e analizza i risultati dei fold (classificazione binaria)
    best_results_dir = os.path.join(save_dir, "best_params", "best_epoch_results")
    all_results = []

    for fold in folds:
        file_path = os.path.join(best_results_dir, f"fold_{fold}", "best_epoch_results.csv")
        if os.path.exists(file_path):
            print(f"📥 Processing results from: {file_path}")
            try:
                df = pd.read_csv(file_path)
                df = df[["epoch", "accuracy", "precision", "recall", "f1", "auc", "score"]]
                df["fold"] = fold
                all_results.append(df)
            except Exception as e:
                print(f"❌ Errore nel leggere {file_path}: {e}")
        else:
            print(f"⚠️ File non trovato: {file_path}")

    if not all_results:
        print("❌ Nessun risultato trovato per i fold!")
        return

    final_df = pd.concat(all_results, ignore_index=True)
    unified_output_file = os.path.join(best_results_dir, "unified_experiment_results.csv")
    final_df.to_csv(unified_output_file, index=False)
    print(f"✅ Dati unificati salvati in {unified_output_file}")

    # === Calcolo media e deviazione standard delle metriche
    metric_columns = ["accuracy", "precision", "recall", "f1", "auc", "score"]
    mean_df = final_df[metric_columns].mean(numeric_only=True).to_frame().T
    std_df = final_df[metric_columns].std(numeric_only=True).to_frame().T

    summary_df = mean_df.copy()
    summary_df.loc[1] = std_df.iloc[0]
    summary_df.index = ["mean", "std"]

    summary_output_file = os.path.join(best_results_dir, "summary_experiment_results.csv")
    summary_df.to_csv(summary_output_file, index=True)
    print(f"✅ Media e deviazione standard salvate in {summary_output_file}")
