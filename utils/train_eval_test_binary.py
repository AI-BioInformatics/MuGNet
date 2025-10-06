import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from utils.loss_visualization import (
    plot_metrics, plot_roc_curve, plot_confusion_matrix
)
from utils.functions import (
    ensure_directory_exists, set_seed, aggregate_attention_scores
)
from utils.loss_visualization import loss_show


def compute_score(auc, f1, precision, recall, accuracy):
    return (0.4 * auc) + (0.2 * f1) + (0.2 * precision) + (0.1 * recall) + (0.1 * accuracy)


def train(
    model, train_loader, val_loader, fold, optimizer,
    device, criterion, save_dir, epochs=50, patience=10, lr_patience=5
):
    set_seed(43)
    model.to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=lr_patience, verbose=True
    )

    best_auc = 0.0
    best_score = 0.0
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_results = {}
    best_attn_scores = None

    attn_train_dir = os.path.join(save_dir, "best_params", "attn_scores", "train", f"fold_{fold}")
    attn_val_dir = os.path.join(save_dir, "best_params", "attn_scores", "val", f"fold_{fold}")
    ensure_directory_exists(attn_train_dir)
    ensure_directory_exists(attn_val_dir)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_attn_scores = []

        for data in train_loader:
            data = data.to(device)
            batch_indices = data.batch.cpu().numpy()
            patient_ids = getattr(data, "patient_id", ["UNKNOWN"] * len(batch_indices))
            patient_ids_per_node = [patient_ids[i] for i in batch_indices]
            tissue_idx = data.tissue_idx.cpu().numpy()

            optimizer.zero_grad()
            out, _, attn_weights, attn_weights1 = model(data.x, data.edge_index, data.batch)
            attn_scores_src, attn_scores_dst, attn_scores1_src, attn_scores1_dst = aggregate_attention_scores(
                data, attn_weights, attn_weights1
            )

            for node_idx in range(len(data.x)):
                train_attn_scores.append({
                    'patient_id': patient_ids_per_node[node_idx],
                    'node_idx': node_idx,
                    'tissue_idx': tissue_idx[node_idx],
                    'attention_score_layer0_src': attn_scores_src[node_idx],
                    'attention_score_layer0_dst': attn_scores_dst[node_idx],
                    'attention_score_layer1_src': attn_scores1_src[node_idx],
                    'attention_score_layer1_dst': attn_scores1_dst[node_idx]
                })

            loss = criterion(out.view(-1), data.y.float().to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        val_result = evaluate(model, val_loader, device, criterion)
        val_loss, accuracy, precision, recall, f1, auc, cm, all_labels, all_probs, all_ids, all_preds, val_attn_scores = val_result
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, AUC = {auc:.4f}")
        scheduler.step(val_loss)

        score = compute_score(auc, f1, precision, recall, accuracy)

        if epoch >= 2:
            update = False
            if auc > best_auc:
                update = True
            elif auc == best_auc and score > best_score:
                update = True
            elif auc >= best_auc - 0.01 and score > best_score + 0.02:
                update = True

            if update:
                best_auc, best_score = auc, score
                best_epoch = epoch + 1
                best_attn_scores = train_attn_scores
                best_results = (accuracy, precision, recall, f1, cm, all_labels, all_probs, all_ids, all_preds)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("🛑 Early stopping attivato!")
                    break

    # Save results
    result_dir = os.path.join(save_dir, "best_params", "best_epoch_results", f"fold_{fold}")
    ensure_directory_exists(result_dir)
    acc, prec, rec, f1, cm, labels, probs, ids, preds = best_results

    pd.DataFrame([{
        "epoch": best_epoch,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": best_auc,
        "score": best_score
    }]).to_csv(os.path.join(result_dir, "best_epoch_results.csv"), index=False)

    if best_attn_scores:
        with open(f"{attn_train_dir}/attn_scores_train_fold_{fold}.pkl", "wb") as f:
            pickle.dump(best_attn_scores, f)
        with open(f"{attn_val_dir}/attn_scores_val_fold_{fold}.pkl", "wb") as f:
            pickle.dump(val_attn_scores, f)

    save_best_model(
        model, fold, best_epoch, best_auc,
        acc, prec, rec, f1, cm, labels, probs, ids, preds,
        base_dir=save_dir
    )

    return train_losses, val_losses


def evaluate(model, data_loader, device, criterion=None):
    model.eval()
    all_labels, all_preds, all_probs, all_ids = [], [], [], []
    val_attn_scores = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out, _, attn_weights, attn_weights1 = model(data.x, data.edge_index, data.batch)
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).long()

            batch_indices = data.batch.cpu().numpy()
            patient_ids = getattr(data, "patient_id", ["UNKNOWN"] * len(batch_indices))
            patient_ids_per_node = [patient_ids[i] for i in batch_indices]
            tissue_idx = data.tissue_idx.cpu().numpy()

            attn_src, attn_dst, attn1_src, attn1_dst = aggregate_attention_scores(data, attn_weights, attn_weights1)

            for node_idx in range(len(data.x)):
                val_attn_scores.append({
                    'patient_id': patient_ids_per_node[node_idx],
                    'node_idx': node_idx,
                    'tissue_idx': tissue_idx[node_idx],
                    'attention_score_layer0_src': attn_src[node_idx],
                    'attention_score_layer0_dst': attn_dst[node_idx],
                    'attention_score_layer1_src': attn1_src[node_idx],
                    'attention_score_layer1_dst': attn1_dst[node_idx]
                })

            all_labels.extend(data.y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_ids.extend(patient_ids if isinstance(patient_ids, list) else [patient_ids])

            if criterion is not None:
                loss = criterion(out.view(-1), data.y.float().to(device))
                total_loss += loss.item()
                n_batches += 1

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    cm = confusion_matrix(all_labels, all_preds)
    avg_loss = total_loss / n_batches if n_batches > 0 else None

    return avg_loss, accuracy, precision, recall, f1, auc, cm, all_labels, all_probs, all_ids, all_preds, val_attn_scores


def test(model, data_loader, device, criterion=None):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    all_ids = []

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out, _, _, _ = model(data.x, data.edge_index, data.batch)
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).long()

            # Patient ID
            batch_indices = data.batch.cpu().numpy()
            patient_ids = getattr(data, "patient_id", ["UNKNOWN"] * len(batch_indices))
            if isinstance(patient_ids, list):
                all_ids.extend(patient_ids)
            else:
                all_ids.append(patient_ids)

            all_labels.extend(data.y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            if criterion is not None:
                loss = criterion(out.view(-1), data.y.float().to(device))
                total_loss += loss.item()
                n_batches += 1

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    cm = confusion_matrix(all_labels, all_preds)
    avg_loss = total_loss / n_batches if n_batches > 0 else None

    return avg_loss, accuracy, precision, recall, f1, auc, cm, all_labels, all_probs, all_ids, all_preds




def save_best_model(model, fold, best_epoch, best_auc, acc, prec, rec, f1, cm, labels, probs, ids, preds, base_dir):
    model_dir = os.path.join(base_dir, "best_params", "best_model", f"fold_{fold}")
    img_dir = os.path.join(base_dir, "best_params", "imgs", f"fold_{fold}", f"evaluate/best_model_epoch_{best_epoch}_auc_{best_auc:.4f}")
    ensure_directory_exists(model_dir)
    ensure_directory_exists(img_dir)

    model_path = os.path.join(model_dir, f"best_model_fold_{fold}.pth")
    torch.save(model.state_dict(), model_path)

    plot_metrics([acc, prec, rec, f1, best_auc], ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'], img_dir)
    plot_roc_curve(labels, probs, img_dir)
    plot_confusion_matrix(cm, img_dir)

    print(f"✅ Miglior modello salvato in: {model_path}")
    print(f"📊 Risultati e immagini salvate in: {img_dir}")
