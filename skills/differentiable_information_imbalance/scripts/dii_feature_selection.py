"""
Differentiable Information Imbalance (DII) for Feature Selection.
Optimizes weights of molecular descriptors to predict nearest neighbors in ground-truth quantum space.

Reference: https://github.com/rangsimanketkaew/mlff
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim

def generate_dii_synthetic_data(num_samples=100, input_dim=8, target_dim=3, noise_dim=5):
    total_input_dim = input_dim + noise_dim
    
    # Ground truth coordinates
    X_clean = torch.randn(num_samples, input_dim)
    Y = torch.matmul(X_clean[:, :target_dim], torch.randn(target_dim, target_dim))
    
    # Add random noise columns to input
    X_noise = torch.randn(num_samples, noise_dim) * 2.0
    X = torch.cat([X_clean, X_noise], dim=1)
    
    return X, Y

def pairwise_distances(x, w=None):
    n = x.size(0)
    if w is not None:
        x = x * w.unsqueeze(0)

    dist_sq = torch.sum(x**2, dim=1, keepdim=True) + torch.sum(x**2, dim=1, keepdim=True).t() - 2.0 * torch.matmul(x, x.t())
    dist_sq = torch.clamp(dist_sq, min=1e-8)
    
    return torch.sqrt(dist_sq)

def compute_soft_ranks(dists, tau=0.1):
    n = dists.size(0)
    dists_expanded_1 = dists.unsqueeze(2) # [n, n, 1]
    dists_expanded_2 = dists.unsqueeze(1) # [n, 1, n]
    diff = dists_expanded_1 - dists_expanded_2
    
    ranks = 1.0 + torch.sum(torch.sigmoid(diff / tau), dim=2)
    
    return ranks

def run_dii_optimization(input_dim, noise_dim, epochs, lr, l1_reg, tau):
    print("Generating synthetic datasets (relevant features: first 3)...")
    X, Y = generate_dii_synthetic_data(num_samples=80, input_dim=input_dim, target_dim=3, noise_dim=noise_dim)
    total_dim = input_dim + noise_dim
    
    # Target space distance metrics and ranks (fixed)
    dists_B = pairwise_distances(Y)
    ranks_B = compute_soft_ranks(dists_B, tau=tau)
    
    w = nn.Parameter(torch.ones(total_dim))
    
    optimizer = optim.Adam([w], lr=lr)
    
    print(f"Optimizing {total_dim} feature weights using DII loss...")
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        w_norm = torch.abs(w)
        
        dists_A = pairwise_distances(X, w_norm)
        
        mask = torch.eye(X.size(0)) * 1e9
        soft_nn = torch.softmax(-(dists_A + mask) / tau, dim=1)
        
        dii_val = torch.mean(torch.sum(soft_nn * ranks_B, dim=1))
        
        l1_penalty = l1_reg * torch.sum(w_norm)
        
        loss = dii_val + l1_penalty
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | DII Value: {dii_val.item():.4f} | L1 Penalty: {l1_penalty.item():.4f} | Total Loss: {loss.item():.4f}")
            
    print("\n--- Feature Optimization Completed ---")

    final_weights = torch.abs(w).detach().numpy()
    
    print("Optimized Feature Weights:")
    for idx, weight in enumerate(final_weights):
        relevance = "Relevant" if idx < 3 else "Noise"
        print(f"  Feature {idx:02d} ({relevance:8s}): Weight = {weight:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Differentiable Information Imbalance (DII)")
    parser.add_argument("--input_dim", type=int, default=5, help="Number of actual clean features")
    parser.add_argument("--noise_dim", type=int, default=5, help="Number of random noise features")
    parser.add_argument("--epochs", type=int, default=50, help="Number of optimization epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--l1_reg", type=float, default=0.05, help="L1 regularization strength")
    parser.add_argument("--tau", type=float, default=0.1, help="Temperature parameter for soft rank/attention")
    
    args = parser.parse_args()
    
    run_dii_optimization(args.input_dim, args.noise_dim, args.epochs, args.lr, args.l1_reg, args.tau)

if __name__ == "__main__":
    main()
