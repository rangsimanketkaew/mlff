#!/usr/bin/env python3

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import LBFGS
from train_pipeline import SimpleAtomicGNN, build_graph_from_structure


class MoleculeInput(BaseModel):
    symbols: list
    positions: list
    cutoff: float = 5.0


class OptimizationInput(BaseModel):
    symbols: list
    positions: list
    cutoff: float = 5.0
    fmax: float = 0.05
    max_steps: int = 100


class MLFFCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, model, cutoff=5.0, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.model.eval()
        self.cutoff = cutoff

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        
        try:
            graph = build_graph_from_structure(
                symbols=symbols,
                positions=positions,
                energy=0.0,
                forces=np.zeros_like(positions),
                cutoff=self.cutoff
            )
            
            x = graph.x
            pos = graph.pos
            edge_index = graph.edge_index
            edge_attr = graph.edge_attr
            
            pos.requires_grad_(True)
            batch = torch.zeros(x.size(0), dtype=torch.long)
            
            energy_pred = self.model(x, pos, edge_index, edge_attr, batch)
            
            forces_pred = -torch.autograd.grad(
                outputs=energy_pred,
                inputs=pos,
                grad_outputs=torch.ones_like(energy_pred),
                retain_graph=False,
                only_inputs=True
            )[0]
            
            self.results['energy'] = float(energy_pred.item())
            self.results['forces'] = forces_pred.detach().cpu().numpy().astype(np.float64)
            
        except Exception:
            self.results['energy'] = -100.0
            self.results['forces'] = np.zeros_like(positions)


app = FastAPI(
    title="MLFF Serving Service",
    description="API endpoints for machine learning force field prediction and structural optimization."
)

GLOBAL_MODEL = None
GLOBAL_DEVICE = torch.device("cpu")


def load_model_from_checkpoint(checkpoint_path):
    model = SimpleAtomicGNN(hidden_dim=64)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


@app.get("/health")
def health_check():
    return {"status": "ready", "device": str(GLOBAL_DEVICE)}


@app.post("/predict")
def predict_properties(input_data: MoleculeInput):
    if GLOBAL_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")
        
    try:
        graph = build_graph_from_structure(
            symbols=input_data.symbols,
            positions=np.array(input_data.positions),
            energy=0.0,
            forces=np.zeros((len(input_data.symbols), 3)),
            cutoff=input_data.cutoff
        )
        
        x = graph.x.to(GLOBAL_DEVICE)
        pos = graph.pos.to(GLOBAL_DEVICE)
        edge_index = graph.edge_index.to(GLOBAL_DEVICE)
        edge_attr = graph.edge_attr.to(GLOBAL_DEVICE)
            
        pos.requires_grad_(True)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=GLOBAL_DEVICE)
        
        energy = GLOBAL_MODEL(x, pos, edge_index, edge_attr, batch)
        forces = -torch.autograd.grad(
            outputs=energy,
            inputs=pos,
            grad_outputs=torch.ones_like(energy),
            only_inputs=True
        )[0]
        
        return {
            "energy": float(energy.item()),
            "forces": forces.detach().cpu().numpy().tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize")
def optimize_geometry(input_data: OptimizationInput):
    if GLOBAL_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")
        
    try:
        atoms = Atoms(
            symbols=input_data.symbols,
            positions=input_data.positions
        )
        
        calc = MLFFCalculator(GLOBAL_MODEL, cutoff=input_data.cutoff)
        atoms.calc = calc
        
        dyn = LBFGS(atoms, logfile=None)
        converged = dyn.run(fmax=input_data.fmax, steps=input_data.max_steps)
        
        return {
            "converged": bool(converged),
            "steps_taken": dyn.get_number_of_steps(),
            "energy": float(atoms.get_potential_energy()),
            "forces": atoms.get_forces().tolist(),
            "optimized_positions": atoms.get_positions().tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model_checkpoint.pt")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    
    args = parser.parse_args()
    
    global GLOBAL_MODEL, GLOBAL_DEVICE
    GLOBAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GLOBAL_MODEL = load_model_from_checkpoint(args.model_path).to(GLOBAL_DEVICE)
    
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
else:
    GLOBAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GLOBAL_MODEL = load_model_from_checkpoint("model_checkpoint.pt").to(GLOBAL_DEVICE)
