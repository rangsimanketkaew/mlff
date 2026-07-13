---
name: delta_ml_transfer_learning
description: Implementing Delta-ML (residual learning between low and high levels of theory) and model transfer learning.
---

# Delta-ML & Transfer Learning

Use this skill when attempting to achieve "chemical accuracy" (~1 kcal/mol) for 3D molecular properties (e.g. system energies, dipole moments, solvation free energy) with limited high-fidelity quantum mechanical or experimental data.

## Conceptual Framework

### Delta-ML ($\Delta$-ML)
Instead of predicting the high-level quantum mechanical (HL) property directly from structure, Delta-ML predicts the *residual difference* between a cheaply calculated low-level (LL) value (e.g., semi-empirical methods like PM7 or low-basis DFT) and the HL value:
$$\Delta = \text{Property}_{\text{HL}} - \text{Property}_{\text{LL}}$$
The ML model is trained to predict $\Delta$. The final prediction is:
$$\text{Property}_{\text{HL\_Predicted}} = \text{Property}_{\text{LL}} + \Delta_{\text{Predicted}}$$
This greatly reduces the learning complexity as the low-level calculation already captures the primary physical interactions.

### Transfer Learning
1. **Pre-training:** Train a neural network model on a large database of cheap calculations (e.g., COSMO-RS or low-level DFT coordinates).
2. **Fine-tuning:** Keep representation backbone weights or adjust with a small learning rate, and fine-tune the output layers on high-quality experimental data.

## Execution

Run the script `delta_transfer_ml.py` to compare standard regression, Delta-ML, and transfer learning:
```bash
python scripts/delta_transfer_ml.py --epochs 30 --lr 0.01
```
This showcases predictions of quantum residuals and fine-tuning transitions.
