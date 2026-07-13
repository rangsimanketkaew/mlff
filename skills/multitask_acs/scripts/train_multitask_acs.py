"""
Multi-Task Learning with Adaptive Checkpointing with Specialization (ACS).
Simulates GNN training on multiple properties and saves specialized task-specific checkpoints.

Reference: https://github.com/rangsimanketkaew/mlff
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def generate_synthetic_data(num_samples=1000, input_dim=32, num_tasks=3):
    X = torch.randn(num_samples, input_dim)
    targets = []
    for i in range(num_tasks):
        w = torch.randn(input_dim, 1) * (i + 1)
        y = torch.matmul(X, w) + torch.randn(num_samples, 1) * 0.1
        targets.append(y)
    Y = torch.cat(targets, dim=1)

    return X, Y

class MultiTaskNet(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, num_tasks=3):
        super().__init__()
        self.shared_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_tasks)
        ])
        
    def forward(self, x):
        shared_feats = self.shared_backbone(x)
        outputs = [head(shared_feats) for head in self.task_heads]

        return torch.cat(outputs, dim=1)

def run_training(num_tasks, hidden_dim, epochs, batch_size, lr, output_dir):
    print("Initializing Multi-Task Network...")
    X, Y = generate_synthetic_data(num_samples=1000, input_dim=32, num_tasks=num_tasks)
    
    train_split = int(0.8 * len(X))
    train_x, val_x = X[:train_split], X[train_split:]
    train_y, val_y = Y[:train_split], Y[train_split:]
    
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)
    
    model = MultiTaskNet(input_dim=32, hidden_dim=hidden_dim, num_tasks=num_tasks)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # ACS state tracking
    best_val_losses = [float('inf')] * num_tasks
    checkpoint_paths = [os.path.join(output_dir, f"task_{i}_best_checkpoint.pt") for i in range(num_tasks)]
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            
            loss = sum(criterion(preds[:, i], batch_y[:, i]) for i in range(num_tasks))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_x)
        
        model.eval()
        task_val_losses = [0.0] * num_tasks
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                for i in range(num_tasks):
                    task_loss = criterion(preds[:, i], batch_y[:, i])
                    task_val_losses[i] += task_loss.item() * batch_x.size(0)
                    
        for i in range(num_tasks):
            task_val_losses[i] /= len(val_x)
            
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Losses: {[round(l, 4) for l in task_val_losses]}")
        
        # Adaptive Checkpointing with Specialization (ACS)
        for i in range(num_tasks):
            if task_val_losses[i] < best_val_losses[i]:
                best_val_losses[i] = task_val_losses[i]
                # Save task-specific weights (shared backbone + task-specific head)
                torch.save({
                    'epoch': epoch,
                    'backbone_state_dict': model.shared_backbone.state_dict(),
                    'head_state_dict': model.task_heads[i].state_dict(),
                    'val_loss': best_val_losses[i]
                }, checkpoint_paths[i])
                print(f"  [ACS Checkpoint] Task {i} achieved new min val loss: {best_val_losses[i]:.4f}. Saved checkpoint.")

    print("\n--- Training Completed ---")
    
    for i in range(num_tasks):
        print(f"Task {i} Best Val Loss: {best_val_losses[i]:.4f} (Saved to {checkpoint_paths[i]})")

def main():
    parser = argparse.ArgumentParser(description="Multi-Task Learning with ACS")
    parser.add_argument("--num_tasks", type=int, default=3, help="Number of concurrent target properties")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size of GNN/MLP backbone")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    run_training(args.num_tasks, args.hidden_dim, args.epochs, args.batch_size, args.lr, args.output_dir)

if __name__ == "__main__":
    main()
