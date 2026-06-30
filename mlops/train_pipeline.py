#!/usr/bin/env python3

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
import mlflow
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import global_add_pool


class SimpleAtomicGNN(nn.Module):
    def __init__(self, num_species=100, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_species, hidden_dim)
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, pos, edge_index, edge_attr, batch):
        h = self.embedding(x)
        row, col = edge_index
        msg = torch.cat([h[row], edge_attr], dim=-1)
        msg = self.msg_mlp(msg)
        
        agg_msg = torch.zeros_like(h)
        agg_msg.index_add_(0, col, msg)
        
        h = self.node_update(torch.cat([h, agg_msg], dim=-1))
        atomic_energies = self.readout(h)
        
        return global_add_pool(atomic_energies, batch)


class DummyAtomicDataset(Dataset):
    def __init__(self, num_samples=100, num_atoms=10):
        from torch_geometric.data import Data
        self.data = []
        for _ in range(num_samples):
            pos = torch.randn(num_atoms, 3)
            x = torch.randint(1, 10, (num_atoms,))
            diff = pos.unsqueeze(1) - pos.unsqueeze(0)
            dists = torch.norm(diff, dim=-1)
            mask = ~torch.eye(num_atoms, dtype=torch.bool)
            edge_index = mask.nonzero().t()
            edge_vectors = diff[edge_index[0], edge_index[1]]
            edge_dists = dists[edge_index[0], edge_index[1]].unsqueeze(1)
            edge_attr = torch.cat([edge_vectors, edge_dists], dim=-1)
            y = torch.tensor([np.random.normal(-500.0, 10.0)], dtype=torch.float32)
            forces = torch.randn(num_atoms, 3)
            
            self.data.append(Data(
                x=x, pos=pos, edge_index=edge_index, 
                edge_attr=edge_attr, y=y, forces=forces
            ))
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]


def compute_energy_and_forces(model, batch_data, device):
    x = batch_data.x.to(device)
    pos = batch_data.pos.to(device)
    edge_index = batch_data.edge_index.to(device)
    edge_attr = batch_data.edge_attr.to(device)
    batch = batch_data.batch.to(device)
    y = batch_data.y.to(device)
    forces_target = batch_data.forces.to(device)

    pos.requires_grad_(True)
    energy_pred = model(x, pos, edge_index, edge_attr, batch)
    
    forces_pred = -torch.autograd.grad(
        outputs=energy_pred,
        inputs=pos,
        grad_outputs=torch.ones_like(energy_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    return energy_pred, forces_pred, y, forces_target


def train(args):
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    
    if is_distributed:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"Distributed training initialized. World size: {world_size}")
        print(f"Device: {device}")
        
    if os.path.exists(os.path.join(args.data_dir, "train_dataset.pt")):
        train_dataset = torch.load(os.path.join(args.data_dir, "train_dataset.pt"))
        val_dataset = torch.load(os.path.join(args.data_dir, "val_dataset.pt"))
    else:
        train_dataset = DummyAtomicDataset(num_samples=200)
        val_dataset = DummyAtomicDataset(num_samples=50)

    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None))
    val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, shuffle=False)

    model = SimpleAtomicGNN(hidden_dim=args.hidden_dim).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse_loss = nn.MSELoss()

    if rank == 0:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("MLFF_Force_Field_Training")
        mlflow.start_run()
        mlflow.log_params({
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "energy_weight": args.energy_weight,
            "force_weight": args.force_weight
        })

    for epoch in range(args.epochs):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        epoch_e_loss = 0.0
        epoch_f_loss = 0.0
        
        start_time = time.time()
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            pred_e, pred_f, target_e, target_f = compute_energy_and_forces(model, batch, device)
            loss_energy = mse_loss(pred_e, target_e)
            loss_forces = mse_loss(pred_f, target_f)
            loss = args.energy_weight * loss_energy + args.force_weight * loss_forces
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_e_loss += loss_energy.item()
            epoch_f_loss += loss_forces.item()

        epoch_loss /= len(train_loader)
        epoch_e_loss /= len(train_loader)
        epoch_f_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_e_loss = 0.0
        val_f_loss = 0.0
        
        with torch.no_grad():
            with torch.enable_grad():
                for batch in val_loader:
                    pred_e, pred_f, target_e, target_f = compute_energy_and_forces(model, batch, device)
                    loss_energy = mse_loss(pred_e, target_e)
                    loss_forces = mse_loss(pred_f, target_f)
                    loss = args.energy_weight * loss_energy + args.force_weight * loss_forces
                    
                    val_loss += loss.item()
                    val_e_loss += loss_energy.item()
                    val_f_loss += loss_forces.item()

        val_loss /= len(val_loader)
        val_e_loss /= len(val_loader)
        val_f_loss /= len(val_loader)

        if rank == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | "
                  f"Train Loss: {epoch_loss:.4f} (E:{epoch_e_loss:.4f}, F:{epoch_f_loss:.4f}) | "
                  f"Val Loss: {val_loss:.4f} (E:{val_e_loss:.4f}, F:{val_f_loss:.4f}) | "
                  f"Time: {elapsed:.2f}s")
                  
            mlflow.log_metrics({
                "train_loss": epoch_loss,
                "train_energy_loss": epoch_e_loss,
                "train_force_loss": epoch_f_loss,
                "val_loss": val_loss,
                "val_energy_loss": val_e_loss,
                "val_force_loss": val_f_loss,
            }, step=epoch)

    if rank == 0:
        model_to_save = model.module if is_distributed else model
        torch.save(model_to_save.state_dict(), args.model_path)
        print(f"Training complete. Model saved to {args.model_path}")
        mlflow.log_artifact(args.model_path)
        mlflow.end_run()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_path", type=str, default="model_checkpoint.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--energy_weight", type=float, default=1.0)
    parser.add_argument("--force_weight", type=float, default=10.0)
    parser.add_argument("--mlflow_uri", type=str, default="http://localhost:5000")
    
    args = parser.parse_args()
    train(args)
