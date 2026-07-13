"""
Deep Ensembles for Uncertainty Quantification in Molecular Property Prediction.
Trains multiple Gaussian neural networks and decomposes prediction uncertainty.

Reference: https://github.com/rangsimanketkaew/mlff
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class GaussianRegressor(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Outputs 2 values: [mean, log_variance]
        self.output_layer = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        features = self.backbone(x)
        out = self.output_layer(features)
        mean = out[:, 0:1]
        log_var = out[:, 1:2]
        log_var = torch.clamp(log_var, min=-8.0, max=4.0)
        return mean, log_var

def nll_loss(mean, log_var, target):
    precision = torch.exp(-log_var)
    squared_error = (target - mean) ** 2
    loss = 0.5 * precision * squared_error + 0.5 * log_var
    return torch.mean(loss)

def generate_uq_data(num_samples=400, input_dim=8):
    # Generate random features
    X = torch.randn(num_samples, input_dim)
    # Target function
    true_w = torch.randn(input_dim, 1)
    base_y = torch.matmul(X, true_w)
    
    noise_scale = 0.2 + 0.8 * torch.abs(X[:, 0:1])
    noise = torch.randn(num_samples, 1) * noise_scale
    Y = base_y + noise
    
    return X, Y, true_w

def train_single_model(X, Y, input_dim, hidden_dim, epochs, lr, seed):
    # random seeds for ensemble diversity
    torch.manual_seed(seed)
    
    model = GaussianRegressor(input_dim=input_dim, hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        mean, log_var = model(X)
        loss = nll_loss(mean, log_var, Y)
        loss.backward()
        optimizer.step()
        
    return model

def main():
    parser = argparse.ArgumentParser(description="Deep Ensembles for Uncertainty Quantification")
    parser.add_argument("--ensemble_size", type=int, default=5, help="Number of models in the ensemble")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs per model")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimensions")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    
    args = parser.parse_args()
    
    input_dim = 8
    X_train, Y_train, true_w = generate_uq_data(num_samples=500, input_dim=input_dim)
    
    print(f"Training ensemble of {args.ensemble_size} models with Gaussian outputs...")
    ensemble = []
    for m in range(args.ensemble_size):
        model = train_single_model(X_train, Y_train, input_dim, args.hidden_dim, args.epochs, args.lr, seed=m*100)
        ensemble.append(model)
        print(f"  Model {m+1}/{args.ensemble_size} trained.")
        
    # Evaluate on a test set containing both in-distribution and out-of-distribution (OOD) samples
    print("\nEvaluating on Test Set...")
    X_test, Y_test, _ = generate_uq_data(num_samples=100, input_dim=input_dim)
    
    # Create OOD test samples (by scaling up features significantly)
    X_test_ood = X_test.clone()
    X_test_ood[:, 0] += 5.0 # Shift first feature to trigger epistemic uncertainty
    
    for label, x_eval, y_eval in [("In-Distribution", X_test, Y_test), ("Out-Of-Distribution (OOD)", X_test_ood, Y_test)]:
        means = []
        variances = []
        
        for model in ensemble:
            model.eval()
            with torch.no_grad():
                pred_mean, pred_log_var = model(x_eval)
                means.append(pred_mean)
                variances.append(torch.exp(pred_log_var))
        
        # Stack predictions
        means = torch.stack(means, dim=0) # [M, N, 1]
        variances = torch.stack(variances, dim=0) # [M, N, 1]
        
        # Calculate ensemble mean
        ensemble_mean = torch.mean(means, dim=0) # [N, 1]
        
        # Calculate Aleatoric uncertainty (average predicted variance)
        aleatoric = torch.mean(variances, dim=0) # [N, 1]
        
        # Calculate Epistemic uncertainty (variance of predicted means)
        epistemic = torch.var(means, dim=0) # [N, 1]
        
        # Calculate Total uncertainty
        total_var = aleatoric + epistemic
        
        # Compute average metrics
        avg_mae = torch.mean(torch.abs(ensemble_mean - y_eval)).item()
        avg_aleatoric = torch.mean(aleatoric).item()
        avg_epistemic = torch.mean(epistemic).item()
        avg_total = torch.mean(total_var).item()
        
        print(f"\n--- Results for {label} ---")
        print(f"  Average Mean Absolute Error (MAE): {avg_mae:.4f}")
        print(f"  Average Aleatoric Uncertainty:   {avg_aleatoric:.4f}")
        print(f"  Average Epistemic Uncertainty:   {avg_epistemic:.4f}")
        print(f"  Average Total Uncertainty:       {avg_total:.4f}")
        
if __name__ == "__main__":
    main()
