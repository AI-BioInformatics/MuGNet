import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
from lifelines.utils import concordance_index
from utils.functions import ensure_directory_exists, set_seed, aggregate_attention_scores
from utils.loss_visualization import save_c_index_plot


def compute_c_index(y, pred, c):
    return concordance_index(y, -pred, c)


def save_best_model(model, fold, best_epoch, c_index_history, best_c_index, base_dir):
    model_save_dir = os.path.join(base_dir, "best_params", "best_model", f"fold_{fold}")
    img_dir = os.path.join(base_dir, "best_params", "imgs", f"fold_{fold}", f"evaluate/best_model_epoch_{best_epoch}")
    ensure_directory_exists(model_save_dir)
    ensure_directory_exists(img_dir)

    torch.save(model.state_dict(), os.path.join(model_save_dir, f"best_model_fold_{fold}.pth"))
    save_c_index_plot(c_index_history, best_epoch, best_c_index, fold, img_dir)
    print(f"✅ Miglior modello salvato in: {model_save_dir}")
    print(f"📊 C-Index plot salvato in: {img_dir}")


def train(model, train_loader, val_loader, fold, optimizer, device, criterion, save_dir, epochs=50, patience=10, lr_patience=5):
    set_seed(43)
    model.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=lr_patience, verbose=True)

    best_c_index = 0.0
    best_epoch = 0
    patience_counter = 0
    train_losses, val_losses = [], []
    best_attn_scores = None
    c_index_history, epoch_results = [], []

    attn_train_dir = os.path.join(save_dir, "best_params", "attn_scores", "train", f"fold_{fold}")
    attn_val_dir = os.path.join(save_dir, "best_params", "attn_scores", "val", f"fold_{fold}")
    ensure_directory_exists(attn_train_dir)
    ensure_directory_exists(attn_val_dir)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_attn_scores = []
        norm_gradients = []

        for data in train_loader:
            data = data.to(device)
            if data.edge_index is None or data.edge_index.size(1) == 0:
                continue

            patient_ids = getattr(data, "patient_id", ["UNKNOWN"] * len(data.batch.cpu().numpy()))
            patient_ids_per_node = [patient_ids[i] for i in data.batch.cpu().numpy()]
            tissue_idx = data.tissue_idx.cpu().numpy()

            optimizer.zero_grad()
            out, _, attn_weights, attn_weights1 = model(data.x, data.edge_index, data.batch)
            attn_src, attn_dst, attn1_src, attn1_dst = aggregate_attention_scores(data, attn_weights, attn_weights1)

            for node_idx in range(len(data.x)):
                train_attn_scores.append({
                    'patient_id': patient_ids_per_node[node_idx],
                    'node_idx': node_idx,
                    'tissue_idx': tissue_idx[node_idx],
                    'attention_score_layer0_src': attn_src[node_idx],
                    'attention_score_layer0_dst': attn_dst[node_idx],
                    'attention_score_layer1_src': attn1_src[node_idx],
                    'attention_score_layer1_dst': attn1_dst[node_idx]
                })

            loss = criterion(out, data.y.to(device), data.c.to(device))
            loss.backward()
            for param in model.parameters():
                if param.grad is not None:
                    norm_gradients.append(param.grad.norm().item())
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        val_loss, c_index, val_attn_scores = evaluate(model, val_loader, device, criterion)
        val_losses.append(val_loss)
        c_index_history.append(c_index)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, C-Index = {c_index:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step(val_loss)

        avg_norm_grad = sum(norm_gradients) / len(norm_gradients) if norm_gradients else 0
        epoch_results.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "c_index": c_index,
            "norm_gradients": avg_norm_grad
        })

        if epoch >= 2 and c_index > best_c_index:
            best_c_index = c_index
            best_epoch = epoch + 1
            best_attn_scores = train_attn_scores
            best_val_attn_scores = val_attn_scores
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("🛑 Early stopping attivato!")
                break

    # Save epoch history
    results_dir = os.path.join(save_dir, "best_params", "results", f"fold_{fold}")
    ensure_directory_exists(results_dir)
    pd.DataFrame(epoch_results).to_csv(os.path.join(results_dir, "epoch_results.csv"), index=False)

    # Save best C-Index
    best_results_dir = os.path.join(save_dir, "best_params", "best_epoch_results", f"fold_{fold}")
    ensure_directory_exists(best_results_dir)
    pd.DataFrame([{"epoch": best_epoch, "c_index": best_c_index}]).to_csv(
        os.path.join(best_results_dir, "best_epoch_results.csv"), index=False
    )

    # Save attention scores
    if best_attn_scores:
        with open(os.path.join(attn_train_dir, f"attn_scores_train_fold_{fold}.pkl"), "wb") as f:
            pickle.dump(best_attn_scores, f)
        with open(os.path.join(attn_val_dir, f"attn_scores_val_fold_{fold}.pkl"), "wb") as f:
            pickle.dump(best_val_attn_scores, f)
        print("✅ Attention scores salvati!")

    save_best_model(model, fold, best_epoch, c_index_history, best_c_index, base_dir=save_dir)

    return train_losses, val_losses


def evaluate(model, data_loader, device, criterion=None):
    model.eval()
    all_y, all_c, all_probs = [], [], []
    val_attn_scores = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            if data.edge_index is None or data.edge_index.size(1) == 0:
                continue

            out, _, attn_weights, attn_weights1 = model(data.x, data.edge_index, data.batch)
            probs = torch.sigmoid(out).cpu().numpy()
            patient_probs = np.max(probs, axis=1)  # Max across bins

            all_probs.extend(patient_probs)
            all_y.extend(data.y.cpu().numpy())
            all_c.extend(data.c.cpu().numpy())

            patient_ids = getattr(data, "patient_id", ["UNKNOWN"] * len(data.batch.cpu().numpy()))
            batch_indices = data.batch.cpu().numpy()
            tissue_idx = data.tissue_idx.cpu().numpy()
            attn_src, attn_dst, attn1_src, attn1_dst = aggregate_attention_scores(data, attn_weights, attn_weights1)

            for node_idx in range(len(data.x)):
                val_attn_scores.append({
                    'patient_id': patient_ids[batch_indices[node_idx]],
                    'node_idx': node_idx,
                    'tissue_idx': tissue_idx[node_idx],
                    'attention_score_layer0_src': attn_src[node_idx],
                    'attention_score_layer0_dst': attn_dst[node_idx],
                    'attention_score_layer1_src': attn1_src[node_idx],
                    'attention_score_layer1_dst': attn1_dst[node_idx]
                })

            if criterion is not None:
                loss = criterion(out, data.y.to(device), data.c.to(device))
                total_loss += loss.item()
                num_batches += 1

    avg_loss = total_loss / num_batches if criterion else None
    c_index = compute_c_index(np.array(all_y), np.array(all_probs), np.array(all_c))
    print(f"C-Index Evaluate: {c_index:.4f}")
    return avg_loss, c_index, val_attn_scores


def test(model, data_loader, device, criterion=None):
    model.eval()
    all_y, all_c, all_probs = [], [], []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            if data.edge_index is None or data.edge_index.size(1) == 0:
                continue

            out, _, _, _ = model(data.x, data.edge_index, data.batch)
            probs = torch.sigmoid(out).cpu().numpy()
            patient_probs = np.max(probs, axis=1)

            all_probs.extend(patient_probs)
            all_y.extend(data.y.cpu().numpy())
            all_c.extend(data.c.cpu().numpy())

            if criterion is not None:
                loss = criterion(out, data.y.to(device), data.c.to(device))
                total_loss += loss.item()
                num_batches += 1

    avg_loss = total_loss / num_batches if criterion else None
    c_index = compute_c_index(np.array(all_y), np.array(all_probs), np.array(all_c))
    print(f"C-Index Test: {c_index:.4f}")
    return avg_loss, c_index, {"C-Index": c_index, "Average Loss": avg_loss}
