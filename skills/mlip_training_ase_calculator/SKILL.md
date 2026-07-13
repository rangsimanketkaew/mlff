---
name: mlip_training_ase_calculator
description: Training Machine Learning Interatomic Potentials (MLIP) to predict energies and forces, and wrapping as an ASE Calculator.
---

# MLIP Training & ASE Calculator Wrapping

Use this skill when developing a custom Machine Learning Interatomic Potential (MLIP) and preparing it for molecular dynamics or structural optimization simulations in Python.

## Conceptual Framework

### Fitting Energies and Forces
Interatomic potentials model the potential energy surface (PES) of a molecular system. To obtain realistic force fields, models must be trained on both total energy $E$ and atomic forces $F$.
- **Forces as Gradients:** Forces are defined physically as the negative gradient of energy with respect to atomic coordinates:
  $$F_i = -\nabla_{R_i} E$$
- **Joint Loss Function:** The loss function balances both predictions:
  $$\text{Loss} = w_E \text{MSE}(E, E_{\text{DFT}}) + w_F \frac{1}{3N} \sum_{i=1}^N \text{MSE}(F_i, F_{i, \text{DFT}})$$

### Wrapping with Atomic Simulation Environment (ASE)
ASE is a standard library for atomic-scale simulations. To run molecular dynamics or structural relaxations, a trained PyTorch model must be wrapped into a custom `ase.calculators.calculator.Calculator`.
The calculator:
1. Receives an `ase.Atoms` object (coordinates and elements).
2. Computes the energy $E(R)$ using a forward pass.
3. Computes forces $F = -\nabla_R E$ using PyTorch Autograd.
4. Returns values in ASE standard format.

## Running the Training and wrapping Script

Run the script `mlip_ase_calc.py` to train a potential on synthetic data and wrap it into an ASE calculator:
```bash
python scripts/mlip_ase_calc.py --epochs 30 --lr 0.01
```
This script demonstrates energy-force fitting and tests the custom calculator.
