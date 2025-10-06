import pandas as pd

# Leggi il file CSV (sostituisci 'path_to_file.csv' con il percorso corretto)
df = pd.read_csv('20_GitHub/outputs/pfi_regression_anat_3bin/splits/patient_labels.csv')

# Conta i pazienti per ogni label
label_counts = df['label'].value_counts()

# Stampa il risultato
print("Numero di pazienti per label:")
print(label_counts)
