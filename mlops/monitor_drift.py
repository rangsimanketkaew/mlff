#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
from scipy.stats import ks_2samp


class StructuralDriftDetector:
    def __init__(self, r_min=0.5, r_max=6.0, num_bins=50):
        self.r_min = r_min
        self.r_max = r_max
        self.num_bins = num_bins
        self.reference_histogram = None
        self.bin_edges = np.linspace(r_min, r_max, num_bins + 1)
        self.min_allowed_distance = 0.75

    def calculate_pairwise_distances(self, positions):
        num_atoms = len(positions)
        if num_atoms < 2:
            return np.array([])
        
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        
        triu_indices = np.triu_indices(num_atoms, k=1)
        return dists[triu_indices]

    def compute_descriptor(self, positions):
        distances = self.calculate_pairwise_distances(positions)
        if len(distances) == 0:
            return np.zeros(self.num_bins)
            
        hist, _ = np.histogram(distances, bins=self.bin_edges)
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist = hist / hist_sum
        return hist

    def fit_reference(self, structures_positions):
        all_dists = []
        for pos in structures_positions:
            all_dists.extend(self.calculate_pairwise_distances(pos))
            
        hist, _ = np.histogram(all_dists, bins=self.bin_edges)
        hist_sum = hist.sum()
        self.reference_histogram = hist / hist_sum if hist_sum > 0 else hist
        print(f"Fit reference detector with {len(structures_positions)} structures.")

    def save_reference(self, file_path):
        if self.reference_histogram is None:
            raise ValueError("No reference distribution to save.")
        data = {
            "r_min": self.r_min,
            "r_max": self.r_max,
            "num_bins": self.num_bins,
            "reference_histogram": self.reference_histogram.tolist()
        }
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved reference distribution to {file_path}")

    def load_reference(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.r_min = data["r_min"]
        self.r_max = data["r_max"]
        self.num_bins = data["num_bins"]
        self.reference_histogram = np.array(data["reference_histogram"])
        self.bin_edges = np.linspace(self.r_min, self.r_max, self.num_bins + 1)
        print(f"Loaded reference distribution from {file_path}")

    def evaluate_structure(self, positions):
        distances = self.calculate_pairwise_distances(positions)
        min_dist = float(distances.min()) if len(distances) > 0 else float('inf')
        
        clash_detected = min_dist < self.min_allowed_distance
        struct_desc = self.compute_descriptor(positions)
        
        if self.reference_histogram is None:
            drift_score = 0.0
            drift_detected = False
        else:
            ref_cdf = np.cumsum(self.reference_histogram)
            struct_cdf = np.cumsum(struct_desc)
            drift_score = float(np.sum(np.abs(ref_cdf - struct_cdf)) / self.num_bins)
            drift_detected = drift_score > 0.15
            
        alert = clash_detected or drift_detected
        
        return {
            "min_distance_angstrom": min_dist,
            "drift_score": drift_score,
            "clash_detected": clash_detected,
            "drift_detected": drift_detected,
            "trigger_active_learning": alert,
            "action": "FLAG_FOR_DFT_RECALCULATION" if alert else "ACCEPT"
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz_file", type=str)
    parser.add_argument("--reference_path", type=str, default="data/drift_reference.json")
    parser.add_argument("--fit_from_dir", type=str)
    
    args = parser.parse_args()
    detector = StructuralDriftDetector()
    
    if args.fit_from_dir:
        import glob
        import torch
        print(f"Fitting reference from preprocessed datasets in {args.fit_from_dir}...")
        positions_list = []
        
        pt_files = glob.glob(os.path.join(args.fit_from_dir, "*_dataset.pt"))
        if pt_files:
            for pt in pt_files:
                dataset = torch.load(pt)
                for data in dataset:
                    if hasattr(data, 'pos'):
                        positions_list.append(data.pos.numpy())
                    elif isinstance(data, dict) and 'pos' in data:
                        positions_list.append(np.array(data['pos']))
        
        if len(positions_list) > 0:
            detector.fit_reference(positions_list)
            os.makedirs(os.path.dirname(args.reference_path), exist_ok=True)
            detector.save_reference(args.reference_path)
        else:
            print("No datasets found to fit reference. Exiting.")
            return

    if args.xyz_file:
        if not os.path.exists(args.reference_path):
            print(f"Reference file {args.reference_path} not found.")
            return
            
        detector.load_reference(args.reference_path)
        print(f"Evaluating structures in {args.xyz_file}...")
        
        from dataset_prep import parse_xyz_manually
        structures = parse_xyz_manually(args.xyz_file)
        
        anomalous_count = 0
        for idx, struct in enumerate(structures):
            res = detector.evaluate_structure(struct['positions'])
            if res['trigger_active_learning']:
                anomalous_count += 1
                print(f"[ALERT] Structure {idx:03d} flags anomalous geometry: "
                      f"Min Dist: {res['min_distance_angstrom']:.2f} Å (Clash: {res['clash_detected']}), "
                      f"Drift Score: {res['drift_score']:.3f} (Drift: {res['drift_detected']}) -> {res['action']}")
                
        print(f"Evaluation complete. Found {anomalous_count} anomalous geometries out of {len(structures)} configuration(s).")
        
    elif not args.fit_from_dir:
        print("Running drift detector dry-run with mock configurations...")
        ref_positions = [np.random.normal(0, 1, (10, 3)) for _ in range(50)]
        detector.fit_reference(ref_positions)
        
        test_pos_normal = np.random.normal(0, 1, (10, 3))
        res_normal = detector.evaluate_structure(test_pos_normal)
        print("Normal test structure evaluation:", json.dumps(res_normal, indent=2))
        
        test_pos_drifted = np.random.normal(0, 1, (10, 3))
        test_pos_drifted[0] = test_pos_drifted[1] + np.array([0.1, 0.05, 0.02])
        res_drifted = detector.evaluate_structure(test_pos_drifted)
        print("Drifted/Clashing structure evaluation:", json.dumps(res_drifted, indent=2))


if __name__ == "__main__":
    main()
