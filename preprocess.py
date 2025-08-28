# preprocess.py (chuẩn hóa cho cả 3 endpoint)
# - Skin Corrosion: MACCS + Physchem → BiLSTM
# - Skin Irritation: MACCS + Physchem → LSTM
# - Skin Sensitization: Token SMILES + RDKit fingerprint → BiLSTM

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdMolDescriptors, AllChem
from tensorflow.keras.preprocessing.sequence import pad_sequences

### ====================================
### 1. CLEANING & CANONICALIZATION
### ====================================

def remove_missing_data(df, smiles_col, label_col):
    return df.dropna(subset=[smiles_col, label_col])

def canonical_smiles(df, smiles_col):
    def canon(sm):
        try:
            mol = Chem.MolFromSmiles(sm)
            return Chem.MolToSmiles(mol, canonical=True) if mol else None
        except:
            return None
    df['canonical_smiles'] = df[smiles_col].apply(canon)
    return df.dropna(subset=['canonical_smiles'])

def remove_inorganic(df, smiles_col):
    def is_organic(sm):
        mol = Chem.MolFromSmiles(sm)
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()] if mol else []
        return 'C' in atoms
    return df[df[smiles_col].apply(is_organic)]

def remove_mixtures(df, smiles_col):
    return df[~df[smiles_col].str.contains('\.')]

def process_duplicate(df, smiles_col, remove_duplicate=True):
    return df.drop_duplicates(subset=[smiles_col]) if remove_duplicate else df


### ====================================
### 2. TOKENIZATION CHO BiLSTM SMILES (sensitization)
### ====================================

def encode_smiles(smiles, tokenizer):
    return tokenizer.encode(smiles)

def process_token_smile_data(df, smiles_col, tokenizer):
    df['encoded'] = df[smiles_col].apply(lambda x: encode_smiles(x, tokenizer))
    return df['encoded'].tolist()

def padding_token(encoded_sequences, max_length=100):
    padded = pad_sequences(encoded_sequences, maxlen=max_length, padding='post')
    return padded

def save_padded_sequences_as_df(padded_sequences, df, file_path):
    df_out = pd.DataFrame(padded_sequences, index=df.index)
    df_out.to_csv(file_path)


### ====================================
### 3. FINGERPRINTS & DESCRIPTORS
### ====================================

def calculate_rdkit(df, smiles_col, nBits=2048):
    def get_rdkit(sm):
        try:
            mol = Chem.MolFromSmiles(sm)
            fp = AllChem.RDKFingerprint(mol)
            return [int(bit) for bit in fp.ToBitString()]
        except:
            return [None] * nBits

    df_fp = df[smiles_col].apply(get_rdkit).apply(pd.Series)
    df_fp.columns = [f'RDKit{i}' for i in range(nBits)]
    return df_fp

def calculate_maccs(df, smiles_col):
    def get_maccs(sm):
        try:
            mol = Chem.MolFromSmiles(sm)
            fp = MACCSkeys.GenMACCSKeys(mol)
            return [int(bit) for bit in fp.ToBitString()]
        except:
            return [None] * 167

    df_fp = df[smiles_col].apply(get_maccs).apply(pd.Series)
    df_fp.columns = [f'MACCS{i}' for i in range(167)]
    return df_fp

def calculate_descriptors(df, smiles_col):
    descriptor_functions = {
        'MolWt': Descriptors.MolWt,
        'LogP': Descriptors.MolLogP,
        'NumHDonors': Descriptors.NumHDonors,
        'NumHAcceptors': Descriptors.NumHAcceptors,
        'TPSA': rdMolDescriptors.CalcTPSA,
        'NumRotatableBonds': Descriptors.NumRotatableBonds,
        'NumAromaticRings': Descriptors.NumAromaticRings,
        'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings,
        'NumHeteroatoms': rdMolDescriptors.CalcNumHeteroatoms,
        'RingCount': rdMolDescriptors.CalcNumRings,
        'HeavyAtomCount': rdMolDescriptors.CalcNumHeavyAtoms,
        'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings
    }

    def get_desc(sm):
        try:
            mol = Chem.MolFromSmiles(sm)
            return [func(mol) for func in descriptor_functions.values()] if mol else [None]*len(descriptor_functions)
        except:
            return [None] * len(descriptor_functions)

    df_desc = df[smiles_col].apply(get_desc).apply(pd.Series)
    df_desc.columns = list(descriptor_functions.keys())
    return df_desc


### ====================================
### 4. ĐẶC TRƯNG CHO CORROSION & IRRITATION
### ====================================

def get_mac_phys_feature(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    maccs = MACCSkeys.GenMACCSKeys(mol)
    physchem = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Descriptors.NumRotatableBonds(mol),
    ]
    maccs_array = [int(bit) for bit in maccs.ToBitString()]
    return np.array(maccs_array + physchem)


### ====================================
### 5. VECTOR COMBINE: TOKEN + RDKit (sensitization)
### ====================================

def get_token_fp_feature(smiles, tokenizer, max_length=100, nBits=2048):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        # Token hóa + padding
        tokens = tokenizer.encode(smiles)
        padded_tokens = pad_sequences([tokens], maxlen=max_length, padding='post')[0]
        # RDKit fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits)
        fp_array = np.array(list(fp.ToBitString())).astype(int)
        # Kết hợp token + fingerprint
        combined = np.concatenate([padded_tokens, fp_array])
        return combined
    except:
        return None
