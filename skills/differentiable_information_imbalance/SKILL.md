---
name: differentiable_information_imbalance
description: Optimizing molecular feature weights and performing feature selection via Differentiable Information Imbalance (DII).
---

# Differentiable Information Imbalance (DII)

Use this skill when performing feature selection or dimension reduction on molecular descriptors or trajectory conformations. This ensures that the chosen feature subset retains maximum predictive power regarding the ground-truth space (e.g. quantum properties or full-atom coordinate systems).

## Conceptual Framework

### Information Imbalance
The Information Imbalance $\Delta(d^A \to d^B)$ measures how well distances in feature space $A$ predict distances in feature space $B$:
$$\Delta(d^A \to d^B) = \frac{2}{N^2} \sum_{i,j: r_{ij}^A = 1} r_{ij}^B$$
where $r_{ij}^A$ and $r_{ij}^B$ represent distance ranks.
- A value near 0 means nearest neighbors in $A$ are also nearest neighbors in $B$ (high predictive power).
- A value near 1 means space $A$ contains no information about space $B$ (random distribution of ranks).

### Differentiable Formulation
To optimize feature weights $w$ in space $A$:
1. We compute weighted distances in $A$: $d_{ij}^A(w) = \sqrt{\sum_k w_k^2 (x_{i,k} - x_{j,k})^2}$.
2. Ranks in the target space $B$ are approximated differentiably using a sigmoid function:
   $$r_{ij}^B \approx 1 + \sum_{k \neq j} \sigma\left(\frac{d_{ij}^B - d_{ik}^B}{\tau}\right)$$
3. The nearest neighbor in $A$ is represented via a soft attention/softmax mapping:
   $$w_{ij}^A = \frac{\exp(-d_{ij}^A(w) / \tau)}{\sum_k \exp(-d_{ik}^A(w) / \tau)}$$
4. The differentiable loss is defined as:
   $$\text{Loss}_{\text{DII}} = \frac{1}{N} \sum_{i=1}^N \sum_{j \neq i} w_{ij}^A \cdot r_{ij}^B + \lambda \|w\|_1$$
   where the $L_1$ penalty encourages feature sparsity (feature selection).

## Execution

Run the script `dii_feature_selection.py` to optimize feature weights for a set of molecular descriptors:
```bash
python scripts/dii_feature_selection.py --lr 0.01 --epochs 50 --l1_reg 0.1
```
This runs gradient descent to identify the most informative descriptors.
