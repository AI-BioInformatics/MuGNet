import os
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from torch import nn
from model import GAT_with_GAP_norm_leakyRELU_06_GraphNorm
from utils.functions import set_seed, ensure_directory_exists, load_graphs_from_folder_by_set_type, init_weights
from utils.loss_visualization import loss_show, save_results_c_index_as_image
from utils.train_eval_test_regr import train, test
from NLL_loss import NLLSurvLoss  # Assicurati che il path sia corretto


def run_regr_training(folds, graphs_dir, save_dir, output_size,
                        embedding_size, num_heads, learning_rate, weight_decay):

    for fold in folds:
        set_seed(43)
        print(f"\n🔹 Fold {fold} in corso... [{fold}/{folds}]\n")

        def load_graphs(set_type):
            folder = os.path.join(graphs_dir, f"fold_{fold}", set_type)
            return load_graphs_from_folder_by_set_type(folder)

        train_graphs = load_graphs("train")
        val_graphs = load_graphs("val")
        test_graphs = load_graphs("test")

        train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

        print(f"Train: {len(train_graphs)} | Val: {len(val_graphs)} | Test: {len(test_graphs)}")

        input_dim = train_graphs[0].x.shape[1]
        print(f"DIM INPUT: {input_dim}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GAT_with_GAP_norm_leakyRELU_06_GraphNorm(
                    input_dim=input_dim,
                    embedding_size=embedding_size,
                    num_heads=num_heads,
                    output_size=output_size
                ).to(device)
        model.apply(init_weights)


        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = NLLSurvLoss(alpha=0.0, reduction="mean")

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
        model_path = os.path.join(model_dir, f"model_fold_{fold}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"✅ Modello salvato in: {model_path}")

        # === Plot loss
        img_dir = os.path.join(save_dir, "best_params", "imgs", f"fold_{fold}", "test")
        ensure_directory_exists(img_dir)
        loss_show(train_losses, val_losses, img_dir)

        # === Evaluation
        model.load_state_dict(torch.load(model_path))
        _, _, results = test(model, test_loader, device, criterion)
        save_results_c_index_as_image(results, img_dir)


    # === Unifica e analizza i risultati dei fold
    best_results_dir = os.path.join(save_dir, "best_params", "best_epoch_results")
    all_results = []

    for fold in folds:
        file_path = os.path.join(best_results_dir, f"fold_{fold}", "best_epoch_results.csv")
        if os.path.exists(file_path):
            print(f"📥 Processing results from: {file_path}")
            try:
                df = pd.read_csv(file_path)
                df = df[["c_index"]]
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

    # === Calcolo media e deviazione standard del C-index
    mean_df = final_df.drop(columns=['fold']).mean(numeric_only=True).to_frame().T
    std_df = final_df.drop(columns=['fold']).std(numeric_only=True).to_frame().T

    summary_df = mean_df.copy()
    summary_df.loc[1] = std_df.iloc[0]
    summary_df.index = ["mean", "std"]

    summary_output_file = os.path.join(best_results_dir, "summary_experiment_results.csv")
    summary_df.to_csv(summary_output_file, index=True)
    print(f"✅ Media e deviazione standard salvate in {summary_output_file}")

