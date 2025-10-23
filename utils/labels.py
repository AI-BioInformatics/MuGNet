import pandas as pd
import os

#################
#       OS      #
#################

# Regression

def os_discrete_balanced(os_file_path, num_bins=3, embedding_dir=None):
    os_df = pd.read_excel(os_file_path)
    os_df = os_df[["patient_id", "OS", "Survival"]]

    # 🔹 If specified, patients are filtered based on available embeddings
    if embedding_dir is not None:
        from os import listdir
        patient_ids_available = set(f.split('_')[0] for f in listdir(embedding_dir) if f.endswith('.pt'))
        os_df = os_df[os_df["patient_id"].isin(patient_ids_available)]
        print(f"🎯 Filtered patients with available embeddings: {len(os_df)}")

    # Creation bin with equal distribution
    os_df["OS_bin"], bin_edges = pd.qcut(os_df["OS"], q=num_bins, retbins=True, labels=range(num_bins))
    print(f"📊OS bin boundaries: {bin_edges}")
    print(f"📈 Counts for each bin:\n{os_df['OS_bin'].value_counts().sort_index()}")

    os_df["c"] = os_df["Survival"].apply(lambda x: 1 if x == 'Alive' else 0)
    os_dict = dict(zip(os_df["patient_id"], zip(os_df["OS_bin"], os_df["c"])))

    return os_dict


# Binary Classification

def os_binary(os_file_path, embedding_dir=None):
    import pandas as pd
    from os import listdir

    # Carica i dati di OS
    os_df = pd.read_excel(os_file_path)
    os_df = os_df[["patient_id", "OS"]]

    # 🔹 Se specificata, filtra i pazienti in base agli embedding disponibili
    if embedding_dir is not None:
        patient_ids_available = set(f.split('_')[0] for f in listdir(embedding_dir) if f.endswith('.pt'))
        os_df = os_df[os_df["patient_id"].isin(patient_ids_available)]
        print(f"🎯 Filtered patients with available embeddings: {len(os_df)}")

    # Calcola la mediana di OS
    #threshold = os_df["OS"].median()
    threshold = 874.50
    print(f"Threshold OS (median): {threshold:.2f}")

    # Binarizza OS (0 = basso, 1 = alto)
    os_df["OS_binary"] = (os_df["OS"] > threshold).astype(int)

    # Stampa il numero di esempi per ciascuna label
    label_counts = os_df["OS_binary"].value_counts()
    print(f"📈 Count for each label:\n{label_counts}")

    # Dizionario {patient_id: OS_binary}
    os_dict = dict(zip(os_df["patient_id"], os_df["OS_binary"]))

    return os_dict




##################
#       PFI      #
##################

# Regression
def pfi_discrete(pfi_file_path, embedding_dir):
    # Carica i dati di PFI
    pfi_df = pd.read_excel(pfi_file_path)
    pfi_df = pfi_df[["patient_id", "PFI", "Survival"]]

    # Rimuove i pazienti con PFI mancante
    initial_count = len(pfi_df)
    pfi_df = pfi_df.dropna(subset=["PFI"])
    removed = initial_count - len(pfi_df)
    if removed > 0:
        print(f"⚠️ Removed {removed} patients with missing PFI")

    # 🔹 Filtra pazienti in base agli embedding disponibili
    embedding_files = os.listdir(embedding_dir)
    available_ids = {fname.split('_')[0] for fname in embedding_files if fname.endswith('.pt')}
    before_filter = len(pfi_df)
    pfi_df = pfi_df[pfi_df["patient_id"].isin(available_ids)]
    filtered_out = before_filter - len(pfi_df)
    print(f"🔎 Removed {filtered_out} patients without embeddings. Remaining: {len(pfi_df)}")

    # Definisci i bin
    bin_edges = [0, 180, 365, float('inf')]
    bin_labels = [0, 1, 2]
    pfi_df["pfi_bin"] = pd.cut(pfi_df["PFI"], bins=bin_edges, labels=bin_labels, include_lowest=True)
    pfi_df = pfi_df.dropna(subset=["pfi_bin"])

    # Calcola la censura
    pfi_df["c"] = pfi_df["Survival"].apply(lambda x: 1 if x == 'Alive' else 0)

    # Info finale
    print(f"📊 Count for each bin:\n{pfi_df['pfi_bin'].value_counts().sort_index()}")

    # Crea dizionario finale
    PFI_dict = dict(zip(pfi_df["patient_id"], zip(pfi_df["pfi_bin"], pfi_df["c"])))

    return PFI_dict



#################
#       HR      #
#################

def hr_binary(labels_path_hr, embedding_dir):
    """
    Returns a dictionary {patient_id: hr_binary} for PDS patients
    with at least 2 out of 3 concordant HR scores (excluding 'conflicting'),
    and for whom embeddings exist.
    hr_binary: 0 = HRD, 1 = HRP
    """
    from collections import Counter

    # === Caricamento e filtro PDS ===
    df_pds = pd.read_csv(labels_path_hr, sep='\t')
    valid_values = ['HRD', 'HRP']

    # === Funzione: almeno due valori uguali e non 'conflicting' ===
    def at_least_two_equal(row):
        values = [row['HRDscarStatus'], row['SBS3status'], row['ID6status']]
        values = [v for v in values if pd.notna(v)]
        if values.count('conflicting') >= 2:
            return False
        filtered = [v for v in values if v != 'conflicting']
        return any(filtered.count(v) >= 2 for v in set(filtered)) if len(filtered) >= 2 else False

    df_filtered = df_pds[df_pds.apply(at_least_two_equal, axis=1)].copy()

    # === Create original_hr (2 out of 3 concordance) ===
    def get_original_hr(row):
        values = [row['HRDscarStatus'], row['SBS3status'], row['ID6status']]
        values = [v for v in values if pd.notna(v)]
        for val in set(values):
            if val != 'conflicting' and values.count(val) >= 2:
                return val
        return None

    df_filtered['original_hr'] = df_filtered.apply(get_original_hr, axis=1)
    df_filtered = df_filtered[df_filtered['original_hr'].isin(valid_values)].copy()

    # === Check patients with embedding ===
    patient_ids = set(df_filtered['patient'].astype(str))
    embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith('.pt')]
    embedding_patient_ids = [fname.split('_')[0] for fname in embedding_files]

    valid_patients = patient_ids & set(embedding_patient_ids)

    # === Final filter
    df_final = df_filtered[df_filtered['patient'].astype(str).isin(valid_patients)].copy()
    df_final['binary_hr'] = df_final['original_hr'].map({'HRP': 1, 'HRD': 0})

    # === Creation dict
    hr_dict = dict(zip(df_final['patient'].astype(str), df_final['binary_hr']))

    return hr_dict
