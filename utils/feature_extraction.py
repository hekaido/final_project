import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem

def mol_dsc_calc(mols):
    return pd.DataFrame({k: f(Chem.MolFromSmiles(m)) for k, f in descriptors.items()} for m in mols)

# список конституционных и физико-химических дескрипторов из библиотеки RDKit
descriptors = {"HeavyAtomCount": Descriptors.HeavyAtomCount,
               "NHOHCount": Descriptors.NHOHCount,
               "NOCount": Descriptors.NOCount,
               "NumHAcceptors": Descriptors.NumHAcceptors,
               "NumHDonors": Descriptors.NumHDonors,
               "NumHeteroatoms": Descriptors.NumHeteroatoms,
               "NumRotatableBonds": Descriptors.NumRotatableBonds,
               "NumValenceElectrons": Descriptors.NumValenceElectrons,
               "NumAromaticRings": Descriptors.NumAromaticRings,
               "NumAliphaticHeterocycles": Descriptors.NumAliphaticHeterocycles,
               "RingCount": Descriptors.RingCount,
               "MW": Descriptors.MolWt,
               "LogP": Descriptors.MolLogP,
               "MR": Descriptors.MolMR,
               "TPSA": Descriptors.TPSA}

def rdkit_fp(smiles_column: pd.Series, radius=3, nBits=2048, useChirality=False):
    # morganFP_rdkit
    def desc_gen(mol):
        mol = Chem.MolFromSmiles(mol)
        bit_vec = np.zeros((1,), np.int16)
        DataStructs.ConvertToNumpyArray(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, useChirality=useChirality), bit_vec)
        return bit_vec

    return pd.DataFrame.from_records(smiles_column.apply(func=desc_gen), columns=[f'bit_id_{i}' for i in range(nBits)])


def rdkit_2d(smiles_column: pd.Series):
    # 2d_rdkit
    descriptors = {i[0]: i[1] for i in Descriptors._descList}
    return pd.DataFrame({k: f(Chem.MolFromSmiles(m)) for k, f in descriptors.items()} for m in smiles_column)

def extract_features(smiles_column: pd.Series):
    X = rdkit_fp(smiles_column)
    Y = rdkit_2d(smiles_column)
    return X, Y