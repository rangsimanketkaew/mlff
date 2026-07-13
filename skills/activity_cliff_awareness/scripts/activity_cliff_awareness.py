"""
Activity Cliff Awareness (ACA) Framework.
Integrates standard regression loss with Triplet Soft Margin (TSM) loss 
on High-Value Activity Cliff Triplets (HV-ACTs).

Reference: https://github.com/rangsimanketkaew/mlff
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim

class ACANet(nn.Module):
    def __init__(self, input_dim=16, embed_dim=8):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )
        self.regressor = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        embeddings = self.backbone(x)
        predictions = self.regressor(embeddings)
        return predictions, embeddings

def mine_triplets(embeddings, inputs, targets, sim_threshold=0.5, act_diff_threshold=1.5):
    """
    Mines High-Value Activity Cliff Triplets (HV-ACTs) from the batch.
    We approximate structural similarity by Euclidean distance between inputs.
    Anchor (a), Positive (p), Negative (n):
    - Similarity(a, p) is HIGH (distance is low), ActivityDiff(a, p) is LOW.
    - Similarity(a, n) is HIGH (distance is low), ActivityDiff(a, n) is HIGH (Activity Cliff).
    """
    n = inputs.size(0)
    
    # Calculate pairwise distances in raw input space (structural similarity)
    dist_struct = torch.cdist(inputs, inputs)
    
    # Calculate pairwise differences in targets
    diff_targets = torch.abs(targets - targets.t())
    
    triplets = []
    for a in range(n):
        for p in range(n):
            if a == p:
                continue
            # Positive: structurally similar (low struct dist) and similar activity
            if dist_struct[a, p] < sim_threshold and diff_targets[a, p] < 0.5:
                for n_idx in range(n):
                    if n_idx == a or n_idx == p:
                        continue
                    if dist_struct[a, n_idx] < sim_threshold and diff_targets[a, n_idx] > act_diff_threshold:
                        triplets.append((a, p, n_idx))
                        
    return triplets

def run_aca_training(epochs, lr, alpha, margin, embed_dim):
    print("Generating synthetic molecular structures and activities...")
    torch.manual_seed(42)
    inputs = torch.randn(100, 16)
    
    true_w = torch.randn(16, 1)
    targets = torch.matmul(inputs, true_w)
    
    inputs[10] = inputs[0] + torch.randn(16) * 0.05
    targets[10] = targets[0] + 3.0
    
    inputs[20] = inputs[5] + torch.randn(16) * 0.05
    targets[20] = targets[5] - 3.0
    
    model = ACANet(input_dim=16, embed_dim=embed_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_criterion = nn.MSELoss()
    
    print("Starting training with Activity Cliff Awareness (ACA)...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        preds, embeddings = model(inputs)
        
        # Standard regression loss
        reg_loss = mse_criterion(preds, targets)
        
        # Mine triplets
        triplets = mine_triplets(embeddings.detach(), inputs, targets)
        
        tsm_loss = torch.tensor(0.0, requires_grad=True)
        if len(triplets) > 0:
            tsm_accum = []
            for a, p, n_idx in triplets:
                h_a = embeddings[a]
                h_p = embeddings[p]
                h_n = embeddings[n_idx]
                
                d_ap = torch.norm(h_a - h_p)
                d_an = torch.norm(h_a - h_n)
                
                # Triplet Soft Margin formula
                loss_val = torch.log(1.0 + torch.exp(d_ap - d_an + margin))
                tsm_accum.append(loss_val)
                
            tsm_loss = torch.mean(torch.stack(tsm_accum))
            
        total_loss = reg_loss + alpha * tsm_loss
        total_loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0 or epoch == 1:
            avg_d_ap = 0.0
            avg_d_an = 0.0
            if len(triplets) > 0:
                with torch.no_grad():
                    d_aps = [torch.norm(embeddings[a] - embeddings[p]).item() for a, p, _ in triplets]
                    d_ans = [torch.norm(embeddings[a] - embeddings[n_idx]).item() for a, _, n_idx in triplets]
                    avg_d_ap = sum(d_aps) / len(d_aps)
                    avg_d_an = sum(d_ans) / len(d_ans)
                    
            print(f"Epoch {epoch:02d} | Reg Loss: {reg_loss.item():.4f} | TSM Loss: {tsm_loss.item():.4f} | Triplets Mined: {len(triplets)} | Avg d_ap: {avg_d_ap:.4f} | Avg d_an: {avg_d_an:.4f}")
            
    print("\n--- Training Completed ---")
    print(f"Final Regression Loss (MSE): {reg_loss.item():.4f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Activity Cliff Awareness (ACA)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight of TSM loss")
    parser.add_argument("--margin", type=float, default=1.0, help="Soft margin parameter")
    parser.add_argument("--embed_dim", type=int, default=8, help="Embedding dimension")
    
    args = parser.parse_args()
    
    run_aca_training(args.epochs, args.lr, args.alpha, args.margin, args.embed_dim)
