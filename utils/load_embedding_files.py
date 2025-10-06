import os
import torch
import numpy as np

def load_embeddings_nact(embeddings_path):
    data = []
    labels_patient = []
    labels_tissue = []
    
    for file in os.listdir(embeddings_path):
        if file.endswith(".pt"):
            filepath = os.path.join(embeddings_path, file)
            embedding = torch.load(filepath, map_location=torch.device('cpu')).numpy()
            
            parts = file.split('_')
            patient_id = parts[0]
            tissue_type = parts[1][1:4]
            
            data.append(embedding)
            labels_patient.append(patient_id)
            labels_tissue.append(tissue_type)
    
    return np.array(data), labels_patient, labels_tissue


'''import re

def extract_tissue_type(filename):
    match = re.search(r'p2?([A-Za-z]{3})', filename)
    if match:
        return match.group(1)
    return 'UNK'  # Unknown

def load_embeddings_pds(embeddings_path):
    data = []
    labels_patient = []
    labels_tissue = []

    for file in os.listdir(embeddings_path):
        if file.endswith(".pt"):
            filepath = os.path.join(embeddings_path, file)
            embedding = torch.load(filepath, map_location=torch.device('cpu')).numpy()

            patient_id = file.split('_')[0]
            tissue_type = extract_tissue_type(file)

            data.append(embedding)
            labels_patient.append(patient_id)
            labels_tissue.append(tissue_type)

    return np.array(data), labels_patient, labels_tissue'''


import os
import torch
import numpy as np
import re
'''
def extract_tissue_type(filename):
    # Cerca prefissi seguiti da almeno 3 lettere
    match = re.search(r'(p2?|r2?|o2?)([A-Za-z0-9]{3,})', filename)
    if match:
        raw_tissue = match.group(2)
        # Normalizza a 'LN' se inizia con 'LN'
        if raw_tissue.upper().startswith('LN'):
            return 'LN'
        return raw_tissue[:3]
    return None'''


import re

def extract_tissue_type(filename):
    # Cerca i prefissi validi: _p, _p2, _r, _r2, _o, _o2 seguiti da un nome tessuto
    match = re.search(r'_(p2?|r2?|o2?)([A-Za-z0-9]+)', filename)
    if match:
        tissue_part = match.group(2)
        if tissue_part.upper().startswith('LN'):
            return 'LN'
        return tissue_part[:3]
    return None





def load_embeddings_pds(embeddings_path):
    data = []
    labels_patient = []
    labels_tissue = []
    unmatched_files = []

    for file in os.listdir(embeddings_path):
        if file.endswith(".pt"):
            filepath = os.path.join(embeddings_path, file)
            embedding = torch.load(filepath, map_location=torch.device('cpu')).numpy()

            patient_id = file.split('_')[0]
            tissue_type = extract_tissue_type(file)

            if tissue_type is None:
                unmatched_files.append(file)
                continue

            data.append(embedding)
            labels_patient.append(patient_id)
            labels_tissue.append(tissue_type)

    if unmatched_files:
        print("❌ File con tipo tessuto non riconosciuto:")
        for fname in unmatched_files:
            print(" -", fname)

    return np.array(data), labels_patient, labels_tissue
