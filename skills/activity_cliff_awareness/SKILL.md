---
name: activity_cliff_awareness
description: Mitigating the effect of activity cliffs using Triplet Soft Margin (TSM) loss on High-Value Activity Cliff Triplets (HV-ACTs).
---

# Activity Cliff Awareness (ACA)

Use this skill when training models for quantitative structure-activity relationship (QSAR) or affinity prediction, where small structural changes can cause massive drops or jumps in biological activity (Activity Cliffs).

## Conceptual Framework

### Activity Cliffs
An **Activity Cliff (AC)** occurs when two molecules are structurally highly similar but have radically different target affinity. Standard machine learning models suffer on these cliff compounds because the representation layers naturally project similar structures near each other, smoothing out the target space.

### The ACA Loss Function
The Activity Cliff Awareness (ACA) framework penalizes smooth representation mapping of cliff compounds. It constructs **High-Value Activity Cliff Triplets (HV-ACTs)**:
- **Anchor ($x_a$):** Target molecule.
- **Positive ($x_p$):** Similar structure, similar activity.
- **Negative ($x_n$):** Similar structure, vastly different activity (the activity cliff).

The combined loss function is:
$$\text{Loss}_{\text{ACA}} = \text{Loss}_{\text{Regression}} + \alpha \cdot \text{Loss}_{\text{TSM}}$$
where $\text{Loss}_{\text{TSM}}$ is the Triplet Soft Margin loss acting on representation embeddings $h$:
$$\text{Loss}_{\text{TSM}} = \ln\left(1 + \exp(d(h_a, h_p) - d(h_a, h_n) + m)\right)$$
where $d$ is a distance metric (e.g., Euclidean) and $m$ is a soft margin.

## Execution

Run the script `activity_cliff_awareness.py` to train a model incorporating TSM loss:
```bash
python scripts/activity_cliff_awareness.py --alpha 0.5 --margin 1.0 --epochs 30
```
This script mines triplets and demonstrates how representation distances change during training.
