---
name: ase_geometry_optimization_md
description: Running molecular geometry optimization and molecular dynamics (MD) simulations using ASE and MLIPs.
---

# Geometry Optimization & Molecular Dynamics Simulations

Use this skill when running relaxation calculations (to find minimum energy structures) or Molecular Dynamics (MD) trajectories using Atomic Simulation Environment (ASE) and a Machine Learning Interatomic Potential (MLIP).

## Conceptual Framework

### Geometry Optimization (Structural Relaxation)
Finding the ground-state conformation requires minimizing the total potential energy of the system with respect to atomic coordinates.
- **L-BFGS Algorithm:** A quasi-Newton method that uses gradients (forces) to iteratively update coordinates, converging to a configuration where forces are close to zero:
  $$F_i = -\nabla_{R_i} E \approx 0$$
- **Max Force Threshold ($f_{max}$):** The relaxation continues until the maximum atomic force component falls below a specified convergence criterion (typically 0.01 to 0.05 eV/Å).

### Molecular Dynamics (MD)
MD simulates the physical movements of atoms over time by solving Newton's equations of motion:
$$F = m \cdot a$$
- **Integration Timestep:** Typically 0.5 to 1.0 femtoseconds (fs) to capture high-frequency molecular vibrations (such as C-H stretches).
- **Thermodynamic Ensembles:**
  - **NVT (Constant Number, Volume, Temperature):** Uses a thermostat (e.g., Langevin or Nose-Hoover) to maintain target temperature by coupling the system to a virtual heat bath.
  - **NPT (Constant Number, Pressure, Temperature):** Uses a barostat to maintain pressure alongside temperature.

## Execution

Run the script `ase_md_simulation.py` to perform optimization and Langevin dynamics:
```bash
python scripts/ase_md_simulation.py --temp 300 --steps 100
```
This script relaxes a cluster of atoms and runs NVT dynamics, saving a `.traj` history.
