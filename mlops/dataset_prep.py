#!/usr/bin/env python3

import os
import argparse
import json
import numpy as np
import torch
import ase
import ase.io

from torch_geometric.data import Data


def parse_xyz_manually(file_path):
    structures = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    idx = 0
    num_lines = len(lines)
    while idx < num_lines:
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        try:
            num_atoms = int(line)
        except ValueError:
            idx += 1
            continue
        
        comment = lines[idx + 1].strip()
        energy = 0.0
        for part in comment.split():
            if part.startswith('energy='):
                try:
                    energy = float(part.split('=')[1].replace('"', ''))
                except ValueError:
                    pass
        
        elements = []
        positions = []
        forces = []
        
        for i in range(num_atoms):
            atom_line = lines[idx + 2 + i].strip().split()
            elements.append(atom_line[0])
            positions.append([float(atom_line[1]), float(atom_line[2]), float(atom_line[3])])
            if len(atom_line) >= 7:
                forces.append([float(atom_line[4]), float(atom_line[5]), float(atom_line[6])])
            else:
                forces.append([0.0, 0.0, 0.0])
                
        structures.append({
            'num_atoms': num_atoms,
            'symbols': elements,
            'positions': np.array(positions, dtype=np.float32),
            'energy': energy,
            'forces': np.array(forces, dtype=np.float32)
        })
        idx += 2 + num_atoms
        
    return structures


def atomic_symbol_to_number(symbol):
    mapping = {
        'H': 1, 
        'He': 2, 
        'Li': 3, 
        'Be': 4, 
        'B': 5, 
        'C': 6, 
        'N': 7, 
        'O': 8, 
        'F': 9, 
        'Ne': 10,
        'Na': 11, 
        'Mg': 12, 
        'Al': 13, 
        'Si': 14, 
        'P': 15, 
        'S': 16, 
        'Cl': 17, 
        'Ar': 18, 
        'K': 19, 
        'Ca': 20
        }
    return mapping.get(symbol.strip(), 1)


def build_graph_from_structure(symbols, positions, energy, forces, cutoff):
    num_atoms = len(symbols)
    atomic_numbers = [atomic_symbol_to_number(s) for s in symbols]
    
    pos_tensor = torch.tensor(positions, dtype=torch.float32)
    diff = pos_tensor.unsqueeze(1) - pos_tensor.unsqueeze(0)
    dists = torch.norm(diff, dim=-1)
    
    mask = (dists < cutoff) & (~torch.eye(num_atoms, dtype=torch.bool))
    edge_index = mask.nonzero().t()
    
    edge_vectors = diff[edge_index[0], edge_index[1]]
    edge_dists = dists[edge_index[0], edge_index[1]].unsqueeze(1)
    
    return Data(
        x=torch.tensor(atomic_numbers, dtype=torch.long),
        pos=pos_tensor,
        edge_index=edge_index,
        edge_attr=torch.cat([edge_vectors, edge_dists], dim=-1),
        y=torch.tensor([energy], dtype=torch.float32),
        forces=torch.tensor(forces, dtype=torch.float32)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--split", type=str, default="0.8,0.1,0.1")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        splits = [float(s) for s in args.split.split(',')]
        assert len(splits) == 3 and sum(splits) == 1.0
    except (ValueError, AssertionError):
        splits = [0.8, 0.1, 0.1]
        
    try:
        atoms_list = ase.io.read(args.input, index=":")
        structures = []
        for atoms in atoms_list:
            energy = atoms.get_potential_energy() if hasattr(atoms, 'get_potential_energy') else 0.0
            try:
                forces = atoms.get_forces()
            except Exception:
                forces = np.zeros((len(atoms), 3))
            
            structures.append({
                'symbols': [atom.symbol for atom in atoms],
                'positions': atoms.positions,
                'energy': energy,
                'forces': forces
            })
    except Exception:
        structures = parse_xyz_manually(args.input)
        
    num_structures = len(structures)
    indices = np.arange(num_structures)
    np.random.shuffle(indices)
    
    train_end = int(splits[0] * num_structures)
    val_end = train_end + int(splits[1] * num_structures)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    splits_dict = {'train': train_idx, 'val': val_idx, 'test': test_idx}
    
    for split_name, idxs in splits_dict.items():
        processed_data = []
        for i in idxs:
            struct = structures[i]
            graph = build_graph_from_structure(
                symbols=struct['symbols'],
                positions=struct['positions'],
                energy=struct['energy'],
                forces=struct['forces'],
                cutoff=args.cutoff
            )
            processed_data.append(graph)
            
        out_path = os.path.join(args.output_dir, f"{split_name}_dataset.pt")
        torch.save(processed_data, out_path)
            
    metadata = {
        "num_structures": num_structures,
        "cutoff_distance_angstrom": args.cutoff,
        "splits": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx)
        }
    }
    with open(os.path.join(args.output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    main()
