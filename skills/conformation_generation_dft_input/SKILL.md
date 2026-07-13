---
name: conformation_generation_dft_input
description: Generating 3D molecular conformations and preparing input files for quantum chemistry DFT calculations.
---

# Conformation Generation & DFT Input Preparation

Use this skill when initializing the computational chemistry pipeline from a 2D representation (SMILES) to obtain 3D coordinates for quantum mechanical calculations (DFT) or Machine Learning Interatomic Potentials (MLIP) input.

## Conceptual Framework

### 3D Conformation Generation
Molecules exist as dynamic ensembles of 3D conformations (conformers) in physical space. For property prediction and force field training, we require realistic 3D geometries.
1. **ETKDG Method:** RDKit's Experimental-Torsion Knowledge Distance Geometry (ETKDG) method is used to generate initial conformer coordinates based on experimental crystal structure databases.
2. **Force Field Relaxation:** Generated conformers are optimized using a fast empirical force field (e.g., MMFF94) to eliminate steric clashes and produce physically realistic starting structures.
3. **Quantum Chemistry Refinement (DFT):** The lowest-energy conformer(s) are selected and exported to higher-fidelity Density Functional Theory (DFT) calculations for ground-truth properties (energies and forces).

## Recommended Workflow

1. **SMILES to 3D Mol:** Convert SMILES to an RDKit Mol object and add hydrogens.
2. **Embed Conformers:** Use `AllChem.EmbedMultipleConfs` with ETKDG parameters.
3. **Minimize Energy:** Use `AllChem.MMFFOptimizeMolecule` to relax the geometries.
4. **Export XYZ:** Extract coordinates of the lowest energy conformer.
5. **Generate Input Deck:** Create an ORCA or Gaussian input file.

## Execution

Run the script `congenerate_dft.py` to generate coordinates and an ORCA input deck for a SMILES string:
```bash
python scripts/congenerate_dft.py --smiles "CCO" --num_conformers 5
```
