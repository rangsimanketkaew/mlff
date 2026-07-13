---
name: deep_ensemble_uq
description: Constructing Gaussian output neural networks and training deep ensembles to quantify aleatoric and epistemic uncertainty.
---

# Deep Ensembles Uncertainty Quantification (UQ)

Use this skill when deploying models for drug discovery screening or active learning loops. This allows the system to recognize out-of-distribution molecules or noisy measurements and compute prediction confidence.

## Conceptual Framework

### Uncertainty Categories
1. **Aleatoric Uncertainty:** Inherent noise in the experimental measurements (e.g., assay noise). Irreducible.
2. **Epistemic Uncertainty:** Model's lack of knowledge due to sparse training data in that chemical region. Reducible by collecting more data.

### Math Formulation
Each model in the ensemble outputs both the predicted mean $\mu(x)$ and variance $\sigma^2(x)$ via a Gaussian output layer. The loss function minimized is the Negative Log-Likelihood (NLL):
$$\text{NLL} = \frac{1}{2} \sum_{k=1}^{N} \left( \ln(\sigma^2(x_k)) + \frac{(y_k - \mu(x_k))^2}{\sigma^2(x_k)} \right)$$

For a query molecule $x$, the total uncertainty is decomposed using $M$ models:
$$\sigma_{\text{total}}^2 = \underbrace{\frac{1}{M} \sum_{m=1}^{M} \sigma_m^2(x)}_{\text{Aleatoric (Inherent Noise)}} + \underbrace{\frac{1}{M} \sum_{m=1}^{M} (\mu_m(x) - \bar{\mu}(x))^2}_{\text{Epistemic (Model Disagreement)}}$$

## Execution

Run the script `deep_ensemble_uq.py` to train an ensemble of Gaussian networks:
```bash
python scripts/deep_ensemble_uq.py --ensemble_size 5 --epochs 20
```
This trains multiple models and decomposes predictive uncertainty for high-error/OOD domains.
