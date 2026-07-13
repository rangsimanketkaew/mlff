"""
ASE Geometry Optimization and Molecular Dynamics (MD) Simulation.
Minimizes potential energy of an atomic cluster using LBFGS, then runs NVT Langevin dynamics.

Reference: https://github.com/rangsimanketkaew/mlff
"""

import argparse
import torch
import torch.nn as nn
from ase import Atoms
from ase import units
from ase.optimize import LBFGS
from ase.md.langevin import Langevin
from ase.calculators.calculator import Calculator, all_properties


class PairwisePotential(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        # Initializing weights so the model produces attractive-repulsive physics 
        # (Lennard-Jones style)
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, positions):
        num_atoms = positions.size(0)
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        dist_sq = torch.sum(diff**2, dim=2)
        mask = ~torch.eye(num_atoms, dtype=torch.bool, device=positions.device)
        dists = torch.sqrt(dist_sq[mask] + 1e-8).unsqueeze(1)
        energies = self.net(dists) + (1.0 / (dists**6 + 1e-3)) - (1.0 / (dists**3 + 1e-3))
        return torch.sum(energies) * 0.5

class MLIPCalculator(Calculator):
    implemented_properties = ['energy', 'forces']
    
    def __init__(self, model, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.model = model
        
    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_properties):
        Calculator.calculate(self, atoms, properties, system_changes)
        
        pos_np = self.atoms.get_positions()
        pos_tensor = torch.tensor(pos_np, dtype=torch.float32, requires_grad=True)
        
        energy_tensor = self.model(pos_tensor)
        
        forces_tensor = torch.autograd.grad(
            outputs=energy_tensor,
            inputs=pos_tensor,
            grad_outputs=torch.ones_like(energy_tensor),
            create_graph=False
        )[0]
        forces_tensor = -forces_tensor
        
        self.results['energy'] = float(energy_tensor.item())
        self.results['forces'] = forces_tensor.detach().numpy().astype(float)

def run_simulation(temp_k, steps, timestep_fs, fmax_opt):
    print("Initializing 4-atom cluster...")
    atoms = Atoms('Ar4', positions=[
        [0.0, 0.0, 0.0],
        [1.8, 0.0, 0.0],
        [0.0, 1.9, 0.0],
        [0.5, 0.5, 2.0]
    ])
    
    model = PairwisePotential()
    atoms.calc = MLIPCalculator(model)
    
    print(f"\n--- Starting Geometry Optimization (LBFGS, fmax={fmax_opt} eV/Angstrom) ---")
    opt = LBFGS(atoms, logfile='-')
    opt.run(fmax=fmax_opt)
    
    print("\nGeometry Optimization completed. Final positions (Angstrom):")
    for idx, pos in enumerate(atoms.get_positions()):
        print(f"  Atom {idx:02d}: {pos}")
        
    print(f"\n--- Starting Molecular Dynamics (Langevin, T={temp_k} K, steps={steps}) ---")
    dt = timestep_fs * units.fs
    
    dyn = Langevin(
        atoms, 
        timestep=dt, 
        temperature_K=temp_k, 
        friction=0.02 / units.fs,
        logfile='-'
    )
    
    def print_energy(a=atoms):
        epot = a.get_potential_energy()
        ekin = a.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB * len(a))
        print(f"    MD Step | Epot: {epot:.4f} eV | Ekin: {ekin:.4f} eV | Temp: {temp:.2f} K")
        
    dyn.attach(print_energy, interval=int(steps / 5))
    
    dyn.run(steps=steps)
    print("\n--- Simulation Completed ---")

def main():
    parser = argparse.ArgumentParser(description="ASE Optimization and MD Simulation")
    parser.add_argument("--temp", type=float, default=300.0, help="Target MD temperature in Kelvin")
    parser.add_argument("--steps", type=int, default=50, help="Number of MD integration steps")
    parser.add_argument("--timestep", type=float, default=1.0, help="Timestep in femtoseconds")
    parser.add_argument("--fmax", type=float, default=0.05, help="Force convergence threshold for relaxation")
    
    args = parser.parse_args()
    
    run_simulation(args.temp, args.steps, args.timestep, args.fmax)

if __name__ == "__main__":
    main()
