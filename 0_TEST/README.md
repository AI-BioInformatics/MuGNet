multi-tissue-gnn/
├── README.md
├── requirements.txt
├── test_pipeline.py
├── best_params.json
├── model/
│   └── model.py
├── utils/
│   ├── functions.py
│   ├── test_utils.py
│   └── load_embeddings.py
├── graphs/
│   ├── classification/
│   │   ├── os/
│   │   │   ├── knn_00/
│   │   │   │   └── fold_1/ ... fold_5/
│   │   │   ├── knn_05/
│   │   │   │   └── fold_1/ ... fold_5/
│   │   │   ├── pca_corr/
│   │   │   │   └── fold_1/ ... fold_5/
│   │   │   └── anat/
│   │   │       └── fold_1/ ... fold_5/
│   │   └── hr/
│   │       ├── knn_00/
│   │       │   └── fold_1/ ... fold_5/
│   │       ├── knn_05/
│   │       │   └── fold_1/ ... fold_5/
│   │       ├── pca_corr/
│   │       │   └── fold_1/ ... fold_5/
│   │       └── anat/
│   │           └── fold_1/ ... fold_5/
│
│   └── regression/
│       ├── os/
│       │   ├── 3bin/
│       │   │   ├── knn_00/
│       │   │   │   └── fold_1/ ... fold_5/
│       │   │   ├── knn_05/
│       │   │   │   └── fold_1/ ... fold_5/
│       │   │   ├── pca_corr/
│       │   │   │   └── fold_1/ ... fold_5/
│       │   │   └── anat/
│       │   │       └── fold_1/ ... fold_5/
│       │   └── 4bin/
│       │       ├── knn_00/
│       │       │   └── fold_1/ ... fold_5/
│       │       ├── knn_05/
│       │       │   └── fold_1/ ... fold_5/
│       │       ├── pca_corr/
│       │       │   └── fold_1/ ... fold_5/
│       │       └── anat/
│       │           └── fold_1/ ... fold_5/
│
│       └── pfi/
│           ├── knn_00/
│           │   └── fold_1/ ... fold_5/
│           ├── knn_05/
│           │   └── fold_1/ ... fold_5/
│           ├── pca_corr/
│           │   └── fold_1/ ... fold_5/
│           └── anat/
│               └── fold_1/ ... fold_5/
├── pre_trained/
│   ├── classification/
│   │   ├── os/
│   │   │   ├── knn_00/
│   │   │   │   ├── model_fold_1.pth
│   │   │   │   └── ...
│   │   │   └── (knn_05, pca_corr, anat)/
│   │   └── hr/
│   │       ├── knn_00/
│   │       └── ...
│
│   └── regression/
│       ├── os/
│       │   ├── 3bin/
│       │   │   └── knn_00/
│       │   │       ├── model_fold_1.pth
│       │   │       └── ...
│       │   ├── 4bin/
│       │   │   └── knn_00/
│       │   │       └── model_fold_1.pth ...
│       └── pfi/
│           ├── knn_00/
│           │   └── model_fold_1.pth ...
│           ├── knn_05/
│           └── ...
