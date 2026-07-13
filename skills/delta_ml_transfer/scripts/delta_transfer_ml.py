"""
Delta-ML and Transfer Learning comparison for molecular properties.
Demonstrates predicting quantum theory differences and fine-tuning pre-trained backbones.

Reference: https://github.com/rangsimanketkaew/mlff
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class PropertyPredictor(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

def generate_delta_dataset(num_samples=500, input_dim=10):
    X = torch.randn(num_samples, input_dim)
    
    # Low-level
    w_ll = torch.randn(input_dim, 1)
    Y_ll = torch.matmul(X, w_ll) + torch.randn(num_samples, 1) * 0.5
    
    # High-level: LL + a non-linear residual
    residual = torch.sin(X[:, 0:1]) * 1.5 + torch.cos(X[:, 1:2]) * 1.0 + torch.randn(num_samples, 1) * 0.1
    Y_hl = Y_ll + residual
    
    return X, Y_ll, Y_hl

def train_model(model, loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
    return model

def main():
    parser = argparse.ArgumentParser(description="Delta-ML & Transfer Learning Simulation")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    
    args = parser.parse_args()
    
    # Generate datasets
    # 1. Large, cheap low-level dataset (for pre-training)
    X_large, Y_ll_large, _ = generate_delta_dataset(num_samples=1000)
    
    # 2. Small, expensive dataset (contains both low-level and high-level labels)
    X_small, Y_ll_small, Y_hl_small = generate_delta_dataset(num_samples=80)
    
    # 3. Test set
    X_test, Y_ll_test, Y_hl_test = generate_delta_dataset(num_samples=200)
    
    criterion = nn.MSELoss()
    mae_fn = nn.L1Loss()
    
    # APPROACH 1: Direct Regression (Baseline)
    # Training directly on small high-level dataset
    print("Running Approach 1: Direct regression on small high-level dataset...")
    direct_model = PropertyPredictor()
    direct_opt = optim.Adam(direct_model.parameters(), lr=args.lr)
    direct_loader = DataLoader(TensorDataset(X_small, Y_hl_small), batch_size=16, shuffle=True)
    direct_model = train_model(direct_model, direct_loader, criterion, direct_opt, args.epochs)
    
    direct_model.eval()
    with torch.no_grad():
        direct_preds = direct_model(X_test)
        mae_direct = mae_fn(direct_preds, Y_hl_test).item()
        
    # APPROACH 2: Delta-ML ($\Delta$-ML)
    # Model learns residual: Delta = Y_hl - Y_ll
    print("Running Approach 2: Delta-ML (residual learning)...")
    delta_targets_small = Y_hl_small - Y_ll_small
    delta_model = PropertyPredictor()
    delta_opt = optim.Adam(delta_model.parameters(), lr=args.lr)
    delta_loader = DataLoader(TensorDataset(X_small, delta_targets_small), batch_size=16, shuffle=True)
    delta_model = train_model(delta_model, delta_loader, criterion, delta_opt, args.epochs)
    
    delta_model.eval()
    with torch.no_grad():
        delta_preds = delta_model(X_test)
        final_hl_preds = Y_ll_test + delta_preds
        mae_delta = mae_fn(final_hl_preds, Y_hl_test).item()
        
    # APPROACH 3: Transfer Learning
    # 3a. Pre-training on large low-level dataset
    print("Running Approach 3: Transfer Learning (Pre-train on LL, fine-tune on HL)...")
    tl_model = PropertyPredictor()
    pretrain_opt = optim.Adam(tl_model.parameters(), lr=args.lr)
    pretrain_loader = DataLoader(TensorDataset(X_large, Y_ll_large), batch_size=32, shuffle=True)
    tl_model = train_model(tl_model, pretrain_loader, criterion, pretrain_opt, epochs=15)
    
    # 3b. Fine-tuning on small high-level dataset (using a smaller learning rate)
    finetune_opt = optim.Adam(tl_model.parameters(), lr=args.lr * 0.1)
    finetune_loader = DataLoader(TensorDataset(X_small, Y_hl_small), batch_size=16, shuffle=True)
    tl_model = train_model(tl_model, finetune_loader, criterion, finetune_opt, epochs=args.epochs)
    
    tl_model.eval()
    with torch.no_grad():
        tl_preds = tl_model(X_test)
        mae_tl = mae_fn(tl_preds, Y_hl_test).item()
        
    print("\n--- Performance Comparison (Mean Absolute Error on Test Set) ---")
    print(f"  Approach 1: Direct Regression (small data):  {mae_direct:.4f}")
    print(f"  Approach 2: Delta-ML (predicting residual): {mae_delta:.4f}")
    print(f"  Approach 3: Transfer Learning (fine-tune):  {mae_tl:.4f}")
    
if __name__ == "__main__":
    main()
