import matplotlib.pyplot as plt
import seaborn as sns
import os

def loss_show(train_losses, val_losses, img_dir):
    plt.figure(figsize=(10, 6))
    
    # Plot delle perdite di training e validazione
    plt.plot(range(len(train_losses)), train_losses, label="Training Loss")
    plt.plot(range(len(val_losses)), val_losses, label="Validation Loss")
    
    # Etichette e titolo
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    # Aggiungi la legenda
    plt.legend()
    
    # Salva il grafico
    plt.savefig(os.path.join(img_dir,"training_loss_plot.png"))
    plt.close()
    
    

def plot_metrics(metrics, metric_names, img_dir):
    """ Funzione per creare un grafico a barre delle metriche """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=metric_names, y=metrics, palette="viridis")
    plt.title("Metriche di valutazione")
    plt.xlabel("Metriche")
    plt.ylabel("Valore")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "metrics_plot.png"))  # Salva il grafico come immagine
    plt.close()

def plot_roc_curve(y_true, y_prob, img_dir):
    """ Funzione per tracciare la ROC Curve """
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "roc_curve.png"))  # Salva la curva ROC come immagine
    plt.close()
    
    
def plot_confusion_matrix(cm, img_dir):
    # Visualizzare la confusion matrix con seaborn
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    
    # Salvare la confusion matrix
    plt.savefig(os.path.join(img_dir, "confusion_matrix.png"))
    plt.close()
    
    
    
def plot_lr_vs_loss(lrs, losses, img_dir):
    """ Funzione per tracciare il grafico dei learning rates vs loss """
    plt.figure(figsize=(10, 6))
    
    # Plot dei learning rate e delle perdite
    plt.plot(lrs, losses)
    
    # Impostazioni per l'asse X in scala logaritmica
    plt.xscale('log')
    
    # Etichette e titolo
    plt.title("Learning Rate vs Loss")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    
    # Salva il grafico come immagine
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "lr_vs_loss_plot.png"))
    plt.close()


def save_c_index_plot(c_index_history, best_epoch, best_c_index, fold, img_dir):
    """
    Salva il grafico del C-Index in un file PNG.

    Parameters:
    - c_index_history (list): La lista che contiene la storia del C-Index per ogni epoca.
    - best_epoch (int): L'epoca che ha ottenuto il miglior C-Index.
    - best_c_index (float): Il valore del C-Index alla miglior epoca.
    - fold (int): Il numero del fold.
    - img_dir (str): La directory in cui salvare l'immagine del grafico.
    """
    # Crea la figura
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(c_index_history) + 1), c_index_history, label='C-Index', color='tab:blue')
    plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Epoch {best_epoch} ({best_c_index:.4f})')
    plt.title(f'C-Index vs Epochs - Fold {fold}')
    plt.xlabel('Epoch')
    plt.ylabel('C-Index')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)  # Griglia più leggera e trasparente
    plt.tight_layout()

    # Salva il grafico
    c_index_plot_path = f"{img_dir}/c_index_plot_fold_{fold}.png"
    plt.savefig(c_index_plot_path)
    plt.close()  # Chiudi il grafico per evitare di sovrascrivere l'immagine
    print(f"📊 Grafico del C-Index salvato in {c_index_plot_path}")


def save_results_c_index_as_image(results, img_dir):
    """
    Salva i risultati (ad esempio, C-Index e Average Loss) come immagine.
    
    Args:
    - results (dict): Dizionario contenente i risultati da visualizzare.
    - img_dir (str): La directory dove salvare l'immagine.
    """
    # Estrai i risultati
    c_index = results.get("C-Index", None)
    avg_loss = results.get("Average Loss", None)

    # Crea il grafico
    plt.figure(figsize=(8, 6))

    # Grafico a barre per visualizzare C-Index e Average Loss
    labels = ['C-Index', 'Average Loss']
    values = [c_index, avg_loss]

    bars = plt.bar(labels, values, color=['blue', 'orange'])
    
    # Aggiungi i valori sopra le barre
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom')

    # Aggiungi le etichette e il titolo
    plt.title("Risultati del Test")
    plt.ylabel("Valori")
    plt.ylim(0, max(values) + 0.1)  # Imposta i limiti per visualizzare bene le barre
    plt.tight_layout()

    # Salva il grafico come immagine
    file_path = f"{img_dir}/test_results.png"
    plt.savefig(file_path)
    plt.close()  # Chiudi la figura per evitare sovrascritture

    print(f"✅ Risultati salvati come immagine in {file_path}")