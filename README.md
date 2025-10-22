# 🧬 MuGNet: A Graph-based Framework for Multi-tissue Integration in Computational Pathology

This repository provides a complete pipeline for building and training a Graph Neural Network (GNN) for survival or biomarker prediction from multi-tissue histological images. Each patient is modeled as a graph, where nodes are tissue-specific Whole Slide Image (WSI) embeddings and edges encode morphological or anatomical relationships.

---

## 🔍 Description

MuGNet is a deep learning framework that models multiple whole-slide images (WSIs) from the same patient as a graph for patient-level prediction tasks in computational pathology. Each WSI is represented as a node, with edges reflecting morphological and biological relationships between tissues. The model leverages Graph Neural Networks (GNNs) with attention-based message passing to integrate and interpret multi-tissue data, enabling accurate and interpretable predictions of clinical outcomes such as survival time and therapy response.

---

## 📊 Dataset information

- **Dataset source**: Derived from the EU DECIDER Project (ClinicalTrials.gov ID: NCT04846933) on high-grade serous ovarian cancer (HGSOC).
- **Data composition**:
  -  243 patients, including NACT and PDS
  -  WSIs from multiple tissue sites (ovary, omentum, peritoneum, mesenterium, lymph nodes, e.g.)

## 🚀 Code information

- **Input**: WSI-level embeddings and clinical labels
- **Output**: Patient graphs, adjacency matrices, and trained models
- **Tasks**:
  - Binary classification (e.g., OS short-term vs long-term)
  - Survival regression (e.g., 3-bin or 4-bin OS, 3-bin PFI)
- **Graph type**: One graph per patient
  - **Nodes**: individual WSIs per patient
  - **Edges**: weighted by similarity metrics between slide embeddings
- **GNN**: 2-layer GAT + Global Attention Pooling

---
## 🏗️ Installation
First, ensure that you have Python and pip installed.
To install all required dependencies, run:
`pip install -r requirements.txt` 

---

## ⚙️ Arguments

| Argument         | Description                                                               |
|------------------|---------------------------------------------------------------------------|
| `--task`         | `classification` or `regression`                                           |
| `--label`        | Clinical label: `os`, `pfi`, or `hr`                                       |
| `--adj_method`   | Adjacency method: `knn_00`, `knn_05`, `anat`, `pca_corr`                  |
| `--num_bins`     | Number of bins for survival regression (ignored for classification)       |
| `--output_dir`   | Output root directory (default: `outputs/`)                                |

---

## 🧾 Input Requirements

- **Labels**:
  - `labels_path`: Excel file with OS / PFI clinical metadata
  - `labels_path_hr`: CSV file with HR status labels
- **Embeddings**:
  - `embeddings_path_nact`: WSI embeddings (TITAN) for OS / PFI (NACT cohort)
  - `embedding_path_pds`: WSI embeddings for HR (PDS cohort)
- **Tissue types**:
  - Supported: `['Adn', 'Per', 'Ome', 'Tub', 'Ova', 'Ute', 'Vag', 'Bow', 'Mes', 'LN']`

---

## 📤 Output Directory Structure

Outputs are saved under: 

```
outputs/{label}_{task}_{adj_method}_{num_bins}bin/

├── adjacency_matrices/
│ └── {adj_method}/
│   ├── train/fold_{i}/tissue_adjacency_matrix_{adj_method}.csv
│   ├── val/fold_{i}/tissue_adjacency_matrix_{adj_method}.csv
│   └── test/fold_{i}/tissue_adjacency_matrix_{adj_method}.csv

├── splits/
│ ├── kfold_patient_splits.csv
│ ├── patient_labels.csv
│ └── split_fold_{i}.csv

├── graphs/
│ └── fold_{i}/
│   ├── train/{patient_id}_graph.pt
│   ├── val/{patient_id}_graph.pt
│   └── test/{patient_id}_graph.pt

├── best_params/
│ ├── attn_scores/
│ │ └── {train,val}/fold_{i}/attn_scores_{set}fold{i}.pkl
│ ├── best_model/
│ │ └── fold_{i}/best_model_fold_{i}.pth
│ ├── model_weights/
│ │ └── model_fold_{i}.pth
│ ├── best_epoch_results/
│ │ ├── fold_{i}/best_epoch_results.csv
│ │ ├── summary_experiment_results.csv
│ │ └── unified_experiment_results.csv
│ ├── pos_weight/
│ │ └── pos_weights_per_fold.csv
│ └── imgs/
│   └──fold_{i}/
│       ├── test/.png
│       └── evaluate/best_model_epoch_/{confusion_matrix.png, metrics_plot.png, roc_curve.png}

```

---

## 🧠 Pipeline Steps

1. **Label Construction**  
   - Binary classification (e.g. OS): `os_binary()`
   - Regression (e.g. OS 3-bin): `os_discrete_balanced()`
   - PFI: `pfi_discrete()`
   - HR: `hr_binary()`

2. **Patient Split Creation**  
   Stratified 5-fold CV with:
   - 70% training
   - 20% validation
   - 10% test  
   Saved in `splits/`.

3. **Adjacency Matrix Generation**  
   Based on `--adj_method`:
   - `knn_00` / `knn_05`: Euclidean similarity in PCA-reduced space
   - `anat`: manually defined anatomical proximity (copied statically)
   - `pca_corr`: correlation across tissue-type means

4. **Patient Graph Generation**  
   Each WSI becomes a node; tissue similarity defines edges.
   Output: PyTorch graphs with features and adjacency.

5. **Model Training**  
   - Binary: `run_binary_training(...)`
   - Regression: `run_regr_training(...)`
   - GNN config:
     - 2× GAT layers
     - Dropout 0.6, LeakyReLU, GraphNorm
     - Global Attention Pooling
     - BCEWithLogitsLoss or NLLSurvLoss

---

## 🧪 Commands

### 🧬 OS – REGRESSION – 3 bins
```bash
python pipeline.py --task regression --label os --adj_method knn_00   --num_bins 3 --output_dir outputs
python pipeline.py --task regression --label os --adj_method knn_05   --num_bins 3 --output_dir outputs
python pipeline.py --task regression --label os --adj_method anat     --num_bins 3 --output_dir outputs
python pipeline.py --task regression --label os --adj_method pca_corr --num_bins 3 --output_dir outputs
```


### 🧬 OS – REGRESSION - 4 bin
```bash
python pipeline.py --task regression --label os --adj_method knn_00   --num_bins 4 --output_dir outputs
python pipeline.py --task regression --label os --adj_method knn_05   --num_bins 4 --output_dir outputs
python pipeline.py --task regression --label os --adj_method anat     --num_bins 4 --output_dir outputs
python pipeline.py --task regression --label os --adj_method pca_corr --num_bins 4 --output_dir outputs
```

### 🧬 PFI - REGRESSION - 3 bin
```bash
python pipeline.py --task regression --label pfi --adj_method knn_00   --num_bins 3 --output_dir outputs
python pipeline.py --task regression --label pfi --adj_method knn_05   --num_bins 3 --output_dir outputs
python pipeline.py --task regression --label pfi --adj_method anat     --num_bins 3 --output_dir outputs
python pipeline.py --task regression --label pfi --adj_method pca_corr --num_bins 3 --output_dir outputs
```

### 🧬 OS - CLASSIFICATION
```bash
python pipeline.py --task classification --label os --adj_method knn_00   --output_dir outputs
python pipeline.py --task classification --label os --adj_method knn_05   --output_dir outputs
python pipeline.py --task classification --label os --adj_method anat     --output_dir outputs
python pipeline.py --task classification --label os --adj_method pca_corr --output_dir outputs
```
### 🧬 HR - CLASSIFICATION
```bash
python pipeline.py --task classification --label hr --adj_method knn_00   --output_dir outputs
python pipeline.py --task classification --label hr --adj_method knn_05   --output_dir outputs
python pipeline.py --task classification --label hr --adj_method anat     --output_dir outputs
python pipeline.py --task classification --label hr --adj_method pca_corr --output_dir outputs
```
