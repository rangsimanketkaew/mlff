---
name: molecular_graph_scaffold_split
description: Molecular graph representations and Bemis-Murcko scaffold splitting to assess OOD generalization.
---

# Molecular Graph & Scaffold Splitting

Use this skill when processing raw molecular data (SMILES or 3D coordinates) and preparing dataset partitions for machine learning training. This ensures realistic validation by avoiding chemical data leakage between splits.

## Conceptual Framework

### Molecular Graph Representation
Molecules are represented as graphs $G = (V, E)$, where:
- Nodes $v \in V$ represent atoms featurized by atomic properties (atomic number, valence, hybridization, formal charge, etc.).
- Edges $e_{uv} \in E$ represent bonds featurized by bond properties (bond type, aromaticity, ring membership).

### Bemis-Murcko Scaffold Splitting
Instead of random splitting, which can lead to over-optimistic performance estimates due to highly similar molecules appearing in both train and test splits, **scaffold splitting** partitions molecules by their core Bemis-Murcko scaffold (the ring structures and their linkers). This ensures:
1. Out-of-distribution (OOD) generalization is evaluated.
2. The model's capacity to generalize to novel chemotypes is measured accurately.

## Recommended Workflow

1. **Preprocess and Clean SMILES:** Use RDKit to standardize structures (desalt, neutralize charges, find canonical SMILES).
2. **Compute Bemis-Murcko Scaffolds:** For each molecule, extract its scaffold.
3. **Group by Scaffold:** Identify all unique scaffolds and group corresponding molecular indices.
4. **Sort and Split:** Sort scaffold groups by size, and assign them to train, validation, and test splits (typically 80/10/10 or 80/20) to ensure that the largest groups are distributed appropriately.
5. **Verify Split:** Check similarity/overlap of chemical space between splits.

## Code Example

Run the helper script `scaffold_split.py` to automatically calculate scaffold splits for a list of SMILES:
```bash
python scripts/scaffold_split.py --input dataset.csv --smiles_col "smiles" --ratios 0.8,0.1,0.1
```
