"""
Scaffold Splitting for Molecular Property Prediction datasets.
Groups molecules by Bemis-Murcko scaffold to ensure out-of-distribution (OOD) testing.

Reference: https://github.com/rangsimanketkaew/mlff
"""

import argparse
import csv
import sys
from collections import defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

DUMMY_SMILES = [
    ("c1ccccc1", 1.2),
    ("c1ccccc1C", 1.5),
    ("c1ccccc1CC", 1.8),
    ("c1ccccc1CCC", 2.1),
    ("c1ccncc1", 0.9),
    ("c1ccncc1C", 1.1),
    ("CC(=O)Oc1ccccc1C(=O)O", 2.5),
    ("Oc1ccccc1C(=O)O", 2.3),
    ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 3.0),
    ("CCO", 0.5),
    ("CC(=O)O", 0.6),
    ("CCCC", 0.8),
]

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return ""

def split_by_scaffold(data, smiles_col, ratios=(0.8, 0.1, 0.1)):
    scaffold_to_indices = defaultdict(list)
    for idx, item in enumerate(data):
        smiles = item[smiles_col] if isinstance(item, dict) else item[0]
        scaffold = get_scaffold(smiles)
        scaffold_to_indices[scaffold].append(idx)
        
    sorted_scaffolds = sorted(
        scaffold_to_indices.items(), 
        key=lambda x: len(x[1]), 
        reverse=True
    )
    
    n_total = len(data)
    train_size = int(ratios[0] * n_total)
    val_size = int(ratios[1] * n_total)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    current_train_size = 0
    current_val_size = 0
    
    for scaffold, indices in sorted_scaffolds:
        if current_train_size + len(indices) <= train_size:
            train_indices.extend(indices)
            current_train_size += len(indices)
        elif current_train_size + current_val_size + len(indices) <= train_size + val_size:
            val_indices.extend(indices)
            current_val_size += len(indices)
        else:
            test_indices.extend(indices)
            
    all_assigned = set(train_indices + val_indices + test_indices)
    for idx in range(n_total):
        if idx not in all_assigned:
            test_indices.append(idx)
            
    return train_indices, val_indices, test_indices, scaffold_to_indices

def main():
    parser = argparse.ArgumentParser(description="Bemis-Murcko Scaffold Splitting")
    parser.add_argument("--input", type=str, help="Path to input CSV file")
    parser.add_argument("--smiles_col", type=str, default="smiles", help="Name of SMILES column")
    parser.add_argument("--ratios", type=str, default="0.8,0.1,0.1", help="Train, val, test ratios comma-separated")
    
    args = parser.parse_args()
    
    try:
        ratios = tuple(float(r) for r in args.ratios.split(","))
        assert len(ratios) == 3 and sum(ratios) == 1.0
    except Exception:
        print("Error: ratios must be three comma-separated values summing to 1.0 (e.g. 0.8,0.1,0.1)")
        sys.exit(1)
        
    data = []
    if args.input:
        try:
            with open(args.input, "r") as f:
                reader = csv.DictReader(f)
                data = list(reader)
            print(f"Loaded {len(data)} molecules from {args.input}")
        except Exception as e:
            print(f"Error reading {args.input}: {e}. Falling back to dummy dataset.")
            data = [dict(zip(["smiles", "property"], item)) for item in DUMMY_SMILES]
    else:
        data = [dict(zip(["smiles", "property"], item)) for item in DUMMY_SMILES]
        
    train_idx, val_idx, test_idx, scaffolds = split_by_scaffold(data, args.smiles_col, ratios)
    
    print("\n--- Scaffold Split Summary ---")
    print(f"Total Molecules: {len(data)}")
    print(f"Unique Scaffolds: {len(scaffolds)}")
    print(f"Train Set: {len(train_idx)} ({len(train_idx)/len(data)*100:.1f}%)")
    print(f"Val Set: {len(val_idx)} ({len(val_idx)/len(data)*100:.1f}%)")
    print(f"Test Set: {len(test_idx)} ({len(test_idx)/len(data)*100:.1f}%)")
    
    print("\nSample Assignments:")
    for split_name, indices in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        print(f"\n{split_name} Molecules:")
        for idx in indices[:3]:
            smiles = data[idx][args.smiles_col]
            sc = get_scaffold(smiles)
            print(f"  SMILES: {smiles} (Scaffold: {sc})")
            
if __name__ == "__main__":
    main()
