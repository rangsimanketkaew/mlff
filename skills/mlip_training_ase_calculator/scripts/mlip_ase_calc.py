"""
MLIP Training and ASE Calculator Wrapping.
Implements a simple pairwise ML potential, trains on energies/forces, and wraps as an ASE Calculator.

Reference: https://github.com/rangsimanketkaew/mlff
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from ase.calculators.calculator import Calculator, all_properties
from ase import Atoms

class PairwisePotential(nn.Module):
    """
    A simple translation/rotation invariant pairwise interatomic potential.
    """
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, positions):
        num_atoms = positions.size(0)
        
        diff = positions.unsqueeze(1) - positions.unsqueeze(0) # [N, N, 3]
        dist_sq = torch.sum(diff**2, dim=2)
        
        mask = ~torch.eye(num_atoms, dtype=torch.bool, device=positions.device)
        dists = torch.sqrt(dist_sq[mask] + 1e-8).unsqueeze(1) # [M, 1]
        
        pair_energies = self.net(dists)
        total_energy = torch.sum(pair_energies) * 0.5
        return total_energy

class MLIPCalculator(Calculator):
    """
    ASE Calculator wrapper for PyTorch-based MLIP models.
    """
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
            create_graph=False,
            retain_graph=False
        )[0]
        forces_tensor = -forces_tensor
        
        self.results['energy'] = float(energy_tensor.item())
        self.results['forces'] = forces_tensor.detach().numpy().astype(float)

def generate_synthetic_pes_data(num_samples=100, num_atoms=4):
    torch.manual_seed(42)
    dataset = []
    
    true_potential = PairwisePotential()
    
    for _ in range(num_samples):
        pos = torch.randn(num_atoms, 3) * 1.5
        pos.requires_grad = True
        
        energy = true_potential(pos)
        
        forces = torch.autograd.grad(energy, pos, grad_outputs=torch.ones_like(energy))[0]
        forces = -forces
        
        dataset.append((pos.detach(), energy.detach(), forces.detach()))
        
    return dataset

def train_mlip(model, dataset, epochs, lr, weight_forces=0.1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    
    print(f"Training MLIP on {len(dataset)} structural snapshots...")
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_energy_loss = 0.0
        epoch_force_loss = 0.0
        
        for pos, true_energy, true_forces in dataset:
            optimizer.zero_grad()
            
            pos_var = pos.clone().requires_grad_(True)
            pred_energy = model(pos_var)
            
            pred_forces = torch.autograd.grad(
                outputs=pred_energy,
                inputs=pos_var,
                create_graph=True,
                retain_graph=True
            )[0]
            pred_forces = -pred_forces
            
            e_loss = mse(pred_energy, true_energy)
            f_loss = mse(pred_forces, true_forces)
            
            loss = e_loss + weight_forces * f_loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_energy_loss += e_loss.item()
            epoch_force_loss += f_loss.item()
            
        epoch_loss /= len(dataset)
        epoch_energy_loss /= len(dataset)
        epoch_force_loss /= len(dataset)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | Combined Loss: {epoch_loss:.6f} | Energy MSE: {epoch_energy_loss:.6f} | Force MSE: {epoch_force_loss:.6f}")
            
    return model

def main():
    parser = argparse.ArgumentParser(description="MLIP Calculator Builder")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train potential")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    
    args = parser.parse_args()
    
    dataset = generate_synthetic_pes_data(num_samples=100, num_atoms=4)
    model = PairwisePotential()
    model = train_mlip(model, dataset, args.epochs, args.lr)
    
    print("\nWrapping model into ASE Calculator and running test inference...")
    calc = MLIPCalculator(model)
    
    atoms = Atoms('H4', positions=[[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
    atoms.calc = calc
    
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    
    print(f"Test Energy: {energy:.6f} eV")
    print("Test Forces (eV/Angstrom):")
    for idx, f in enumerate(forces):
        print(f"  Atom {idx}: {f}")

if __name__ == "__main__":
    main()
