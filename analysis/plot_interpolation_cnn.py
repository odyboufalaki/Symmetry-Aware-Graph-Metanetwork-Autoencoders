#!/usr/bin/env python3
"""
Standalone plotter for model interpolation curves.

This script loads pre-computed result dictionaries from a fixed directory
and plots the specified metric (accuracy or loss) for multiple experiments
on the same axes.

It loads data from:
- analysis/resources/interpolation/matrices/cnn/interpolation_results.pt
- analysis/resources/interpolation/matrices/cnn/interpolation_results_ng.pt

Example Usage:
  # Plot the accuracy curves for all experiments
  python analysis/plot_interpolation_cnn.py --metric accuracies

  # Plot the loss curves for only the autoencoders
  python analysis/plot_interpolation_cnn.py --experiments scalegmn neural_graphs --metric losses
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for saving files
import matplotlib.pyplot as plt

# Assumes this utility file exists for styling, as requested
from analysis.utils import plotting as plt_utils


# ------------------------------
# Configuration & Mappings
# ------------------------------

# Hardcoded directories, as requested
MATRICES_DIR = Path("analysis/resources/interpolation/cnn")
OUTPUT_DIR = Path("analysis/resources/interpolation/cnn")

# Maps CLI experiment name to a pretty label for the plot legend
LabelMap: Dict[str, str] = {
    "original": "Original Space",
    "rebasin": "Linear Assignment (Re-Basin)",
    "scalegmn": "Autoencoder (ScaleGMN)",
    "neural_graphs": "Autoencoder (Neural Graphs)",
}

# Maps CLI experiment name to its source file and the key within that file.
# If the key is `None`, the entire file's content is the result dictionary.
FileKeyMap: Dict[str, Tuple[str, Optional[str]]] = {
    "original": ("interpolation_results.pt", "Original Space"),
    "rebasin": ("interpolation_results.pt", "Linearly Aligned Space"),
    "scalegmn": ("interpolation_results.pt", "Reconstructed Space"),
    "neural_graphs": ("interpolation_results_ng.pt", None),
}

# Maps the final plot label to a color specification for plt_utils.flexoki
MethodColors: Dict[str, Tuple[str, int]] = {
    "Original Space": ("Red", 500),
    "Autoencoder (ScaleGMN)": ("Yellow", 500),
    "Linear Assignment (Re-Basin)": ("Cyan", 500),
    "Autoencoder (Neural Graphs)": ("Purple", 500),
    "Latent space": ("Orange", 500),
}


# ------------------------------
# CLI Parser
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Plotter for model interpolation curves.")
    p.add_argument(
        "--experiments",
        nargs='+',
        default=list(LabelMap.keys()),
        help=f"Space-separated list of experiments to plot. Default: all. Allowed: {list(LabelMap.keys())}"
    )
    p.add_argument(
        "--metric",
        type=str,
        required=True,
        choices=["accuracies", "losses"],
        help="The metric to plot from the saved result files.",
    )

    p.add_argument(
        "--band",
        type=str,
        default="sem",
        choices=["sem", "std", "none"],
        help="Type of error band to plot ('sem', 'std', or 'none')."
    )
    return p.parse_args()


def _curve_stats(matrix: np.ndarray):
    """
    Calculates statistics for a matrix of curves.
    matrix: [N, T] where N is num runs, T is num interpolation steps.
    Returns mean, std, sem, N.
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected matrix [N, T], got {tuple(matrix.shape)}")
    mean = matrix.mean(axis=0)
    std  = matrix.std(axis=0)
    n    = max(int(matrix.shape[0]), 1)
    sem  = std / np.sqrt(n)
    return mean, std, sem, n


# ------------------------------
# Helper & Plotting Functions
# ------------------------------
def _resolve_color(label: str):
    """Resolves a plot label to a specific color using plt_utils."""
    spec = MethodColors.get(label)
    if spec is None:
        return None  # Use default color cycle
    name, shade = spec
    return plt_utils.flexoki(name, shade)

def plot_curves(
    results: list,
    metric_name: str,
    band: str,
    save_path: Path,
):
    """Generates and saves a plot of the interpolation curves with error bands."""
    if not results:
        raise ValueError("No results to plot.")

    # Infer T from the first valid data array
    T = len(results[0][0])
    xs = np.linspace(0.0, 1.0, T)

    
    plt_utils.set_flexoki_cycle()
    plt.figure(figsize=(7.0, 4.5), dpi=150)

    for mean_curve, raw_matrix, label in results:
        color = _resolve_color(label)
        plt.plot(xs, mean_curve, marker='.', linestyle='-', linewidth=3.0, label=label, color=color)

        if raw_matrix is not None and band != "none":
            _, std, sem, n = _curve_stats(raw_matrix)
            spread = sem if band == "sem" else std
            lower, upper = mean_curve - spread, mean_curve + spread
            
            print(f"[info] Plotting '{label}' (N={n}): mean={mean_curve.mean():.4f}, std={std.mean():.4f}, sem={sem.mean():.4f}")
            plt.fill_between(xs, lower, upper, alpha=0.30, linewidth=0, color=color)

    # Highlight endpoints using data from the 'Original Space' if available
    original_mean = next((mean for mean, raw, label in results if label == "Original Space"), None)
    if original_mean is not None:
        plt.scatter(
            [0, 1], [original_mean[0], original_mean[-1]],
            s=150, marker='*', c='black', zorder=5, label='Endpoint Models'
        )

    plt.xlabel("Interpolation $t$", fontsize=28)
    y_label = "Accuracy" if metric_name == "accuracies" else "Loss"
    plt.ylabel(y_label, fontsize=28)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


# ------------------------------
# Main Execution Block
# ------------------------------
def main():
    args = parse_args()
    
    # Will store tuples of (mean_curve, raw_matrix, label)
    loaded_results = []

    print("--- Loading Experiment Results ---")
    for exp_key in args.experiments:
        if exp_key not in FileKeyMap:
            print(f"[warning] Experiment '{exp_key}' is not recognized. Skipping.", file=sys.stderr)
            continue

        filename, dict_key = FileKeyMap[exp_key]
        file_path = MATRICES_DIR / filename

        if not file_path.exists():
            print(f"[error] File not found: {file_path}. Skipping '{exp_key}'.", file=sys.stderr)
            continue

        data = torch.load(file_path, map_location="cpu")
        
        experiment_results = data.get(dict_key) if dict_key else data
        if experiment_results is None:
            print(f"[error] Key '{dict_key}' not found in {filename}. Skipping '{exp_key}'.", file=sys.stderr)
            continue

        # Load mean curve (must exist)
        mean_data = experiment_results.get(args.metric)
        if mean_data is None:
            print(f"[error] Metric '{args.metric}' not in results for '{exp_key}'. Skipping.", file=sys.stderr)
            continue
        
        # Load raw data matrix (optional)
        raw_metric_key = f"raw_{args.metric}"
        raw_data = experiment_results.get(raw_metric_key)
        
        mean_curve_np = np.array(mean_data)
        raw_matrix_np = np.array(raw_data) if raw_data is not None else None
        
        label = LabelMap[exp_key]
        
        loaded_results.append((mean_curve_np, raw_matrix_np, label))
        
        raw_info = f"shape={raw_matrix_np.shape}" if raw_matrix_np is not None else "no raw data found"
        print(f"[info] Loaded '{label}' ({args.metric}), {raw_info}")

    if not loaded_results:
        print("\n[error] No data was loaded. Nothing to plot.", file=sys.stderr)
        sys.exit(1)

    output_filename = f"interpolation_curves_{args.metric}_band_{args.band}.pdf"
    output_path = OUTPUT_DIR / output_filename
    
    plot_curves(
        results=loaded_results,
        metric_name=args.metric,
        band=args.band,
        save_path=output_path,
    )

    print(f"\n[ok] Saved plot -> {output_path}")

if __name__ == "__main__":
    main()