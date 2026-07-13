"""
Active Learning (AL) Loop simulation for Machine Learning Interatomic Potentials.
Simulates running exploratory MD, calculating force variance, querying DFT, and retraining.

Reference: https://github.com/rangsimanketkaew/mlff
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim

class PairwisePotential(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
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
        
        energies = self.net(dists) + (2.0 / (dists**6 + 1e-3)) - (2.0 / (dists**3 + 1e-3))
        return torch.sum(energies) * 0.5

class DFTOracle:
    """
    Simulates a DFT calculation (evaluates ground truth potential energy and gradients).
    """
    def __init__(self):
        self.true_pot = PairwisePotential()
        with torch.no_grad():
            for p in self.true_pot.parameters():
                p.add_(torch.randn_like(p) * 0.2)
                
    def calculate(self, positions):
        pos = positions.clone().requires_grad_(True)
        energy = self.true_pot(pos)
        forces = -torch.autograd.grad(energy, pos)[0]
        return energy.detach(), forces.detach()

def train_ensemble(ensemble, dataset, epochs=15, lr=0.01):
    mse = nn.MSELoss()
    for idx, model in enumerate(ensemble):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        torch.manual_seed(idx * 100)
        
        for epoch in range(epochs):
            for pos, true_energy, true_forces in dataset:
                optimizer.zero_grad()
                pos_var = pos.clone().requires_grad_(True)
                pred_energy = model(pos_var)
                pred_forces = -torch.autograd.grad(pred_energy, pos_var, create_graph=True)[0]
                
                loss = mse(pred_energy, true_energy) + 0.1 * mse(pred_forces, true_forces)
                loss.backward()
                optimizer.step()

def evaluate_test_set(ensemble, test_set):
    mse = nn.MSELoss()
    total_e_mae = 0.0
    total_f_mae = 0.0
    
    for pos, true_energy, true_forces in test_set:
        pred_energies = []
        pred_forces = []
        for model in ensemble:
            model.eval()
            pos_var = pos.clone().requires_grad_(True)
            pred_e = model(pos_var)
            pred_f = -torch.autograd.grad(pred_e, pos_var)[0]
            pred_energies.append(pred_e)
            pred_forces.append(pred_f)
            
        avg_energy = torch.mean(torch.stack(pred_energies))
        avg_forces = torch.mean(torch.stack(pred_forces), dim=0)
        
        total_e_mae += torch.mean(torch.abs(avg_energy - true_energy)).item()
        total_f_mae += torch.mean(torch.abs(avg_forces - true_forces)).item()
        
    return total_e_mae / len(test_set), total_f_mae / len(test_set)

def run_active_learning(num_iterations, query_threshold, num_exploratory):
    print("Initializing Simulated DFT Oracle...")
    oracle = DFTOracle()
    
    # 1. Initialize Seed Dataset (only 5 structure snapshots)
    print("Generating initial training seed dataset (5 structures)...")
    train_dataset = []
    for _ in range(5):
        pos = torch.randn(4, 3) * 1.5
        energy, forces = oracle.calculate(pos)
        train_dataset.append((pos, energy, forces))
        
    test_set = []
    for _ in range(50):
        pos = torch.randn(4, 3) * 1.5
        energy, forces = oracle.calculate(pos)
        test_set.append((pos, energy, forces))
        
    # 2. Instantiate Model Ensemble (3 members)
    ensemble = [PairwisePotential() for _ in range(3)]
    
    for cycle in range(1, num_iterations + 1):
        print(f"\n================ Active Learning Cycle {cycle}/{num_iterations} ================")
        print(f"Dataset Size: {len(train_dataset)} structures")
        
        train_ensemble(ensemble, train_dataset, epochs=20, lr=0.01)
        
        test_e_mae, test_f_mae = evaluate_test_set(ensemble, test_set)
        print(f"  Current Test Accuracy -> Energy MAE: {test_e_mae:.4f} eV | Force MAE: {test_f_mae:.4f} eV/A")
        
        exploratory_pool = [torch.randn(4, 3) * 1.5 for _ in range(num_exploratory)]
        
        # Compute force predictions and uncertainty (std dev of predicted forces)
        queried_structures = []
        uncertainties = []
        
        for pos in exploratory_pool:
            member_forces = []
            for model in ensemble:
                model.eval()
                pos_var = pos.clone().requires_grad_(True)
                pred_e = model(pos_var)
                pred_f = -torch.autograd.grad(pred_e, pos_var)[0]
                member_forces.append(pred_f)
                
            member_forces = torch.stack(member_forces)
            force_std = torch.std(member_forces, dim=0) # [N, 3]
            max_force_std = torch.max(force_std).item()
            
            if max_force_std > query_threshold:
                queried_structures.append(pos)
                uncertainties.append(max_force_std)
                
        print(f"  Explored {num_exploratory} frames | Found {len(queried_structures)} structures exceeding uncertainty threshold ({query_threshold})")
        
        if len(queried_structures) == 0:
            print("  No high-uncertainty structures found. Convergence achieved! Ending loop.")
            break
            
        sorted_queries = sorted(zip(queried_structures, uncertainties), key=lambda x: x[1], reverse=True)
        to_calculate = sorted_queries[:5]
        
        print(f"  Sending {len(to_calculate)} structures to DFT Oracle...")
        for pos, std in to_calculate:
            energy, forces = oracle.calculate(pos)
            train_dataset.append((pos, energy, forces))
            
    train_ensemble(ensemble, train_dataset, epochs=20, lr=0.01)
    final_e_mae, final_f_mae = evaluate_test_set(ensemble, test_set)
    
    print("\n================ Active Learning Summary ================")
    print(f"Final Dataset Size: {len(train_dataset)} structures")
    print(f"Final Test Accuracy -> Energy MAE: {final_e_mae:.4f} eV | Force MAE: {final_f_mae:.4f} eV/A")

def main():
    parser = argparse.ArgumentParser(description="Active Learning Loop Simulation")
    parser.add_argument("--al_iterations", type=int, default=3, help="Number of active learning query cycles")
    parser.add_argument("--query_threshold", type=float, default=0.5, help="Standard deviation threshold for force query")
    parser.add_argument("--num_exploratory", type=int, default=30, help="Number of exploratory coordinates generated per cycle")
    
    args = parser.parse_args()
    
    run_active_learning(args.al_iterations, args.query_threshold, args.num_exploratory)

if __name__ == "__main__":
    main()
