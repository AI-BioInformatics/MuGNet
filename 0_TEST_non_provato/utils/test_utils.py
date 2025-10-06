from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch
import numpy as np
from utils.functions import compute_c_index  # assicurati sia disponibile

def evaluate_model(model, loader, criterion, device, task):
    model.eval()
    losses = []

    if task == "classification":
        all_labels = []
        all_preds = []
        all_probs = []
        all_ids = []

        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out, _, _, _ = model(data.x, data.edge_index, data.batch)
                probs = torch.sigmoid(out.view(-1))
                preds = (probs > 0.5).long()

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
                    losses.append(loss.item())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
        cm = confusion_matrix(all_labels, all_preds)
        avg_loss = sum(losses) / len(losses) if losses else None

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc
        }

    elif task == "regression":
        all_y, all_c, all_probs = [], [], []

        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out, _, _, _ = model(data.x, data.edge_index, data.batch)
                probs = np.max(torch.sigmoid(out).cpu().numpy(), axis=1)
                all_probs.extend(probs)
                all_y.extend(data.y.cpu().numpy())
                all_c.extend(data.c.cpu().numpy())

                if criterion is not None:
                    loss = criterion(out, data.y.to(device), data.c.to(device))
                    losses.append(loss.item())

        avg_loss = sum(losses) / len(losses) if losses else None
        c_index = compute_c_index(np.array(all_y), np.array(all_probs), np.array(all_c))

        return {
            "loss": avg_loss,
            "c_index": c_index
        }

    else:
        raise ValueError("Task must be either 'classification' or 'regression'")
