# Foundation Machine Learning Force Field (MLFF) at Scale

Repository for MLOps pipeline for training, deploying, and monitoring MLFFs for chemistry and drug discovery.

*The terms Machine Learning Force Field (MLFF) and Machine Learning Interatomic Potential (MLIP) are used interchangeably.*

## MLFF for (Quantum) Chemistry and Drug Discovery

Foundation models such as **UMA** (Universal Models for Atoms) and **MACE** (Message Passing Atomic Cluster Expansion) use massive pre-training datasets to capture complex, multi-body interactions and physical symmetries of molecules. In drug discovery, these pre-trained potentials allow for rapid, high-fidelity geometry optimizations, conformer searches, and molecular dynamics simulations of drug-target complexes without requiring system-specific retraining.

## 🚀 MLOps Framework & Resources

Hands-on guide for training, deploying, and monitoring MLFF model that scales automatically.

### 📖 [MLOps Comprehensive Guide](mlops/README.md)

The guide covers

1. **Lifecycles** - Active learning loop and data verification.
2. **Scaling** - Distributed training architectures for HPC (SLURM) and AWS Cloud (SageMaker, FSx for Lustre).
3. **Serving** - High-throughput FastAPI and Triton Inference Server wrapping.
4. **Monitoring** - Real-time out-of-distribution (OOD) geometry & bond-clash detection.

### 🛠️ MLOps Core Scripts

Our pipeline consists of the following components under the [mlops/](mlops) folder

* **[dataset_prep.py](mlops/dataset_prep.py)**: Converts `.xyz`/`.extxyz` coordinates into PyTorch Geometric graph datasets based on distance cutoffs.
* **[train_pipeline.py](mlops/train_pipeline.py)**: Distributed DDP training script in PyTorch that computes energies and derives forces analytically using double autograd. Logs metrics to MLflow.
* **[inference_service.py](mlops/inference_service.py)**: FastAPI microservice exposing `/predict` (energies and forces) and `/optimize` (structure relaxation integrating an ASE LBFGS optimizer).
* **[monitor_drift.py](mlops/monitor_drift.py)**: Detects atomic bond clashes and checks geometric drift using pairwise distance distributions to alert on OOD structures.

### 🏢 Orchestration & Infrastructure Templates

* **[submit_hpc.sh](mlops/submit_hpc.sh)**: A SLURM submit template for multi-node, multi-GPU training clusters via `torchrun`.
* **[run_sagemaker.py](mlops/run_sagemaker.py)**: AWS SageMaker SDK launcher targeting large multi-GPU instances (e.g. `ml.p4d.24xlarge`) utilizing FSx for Lustre.

### 👨‍💻 Author

[Rangsiman Ketkaew](https://rangsimanketkaew.github.io/) <br>
ML PostDoc Researcher, ETH Zurich, Switzerland
