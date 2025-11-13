import argparse
import glob
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from analysis.utils import plotting as plt_utils  # has set_flexoki_cycle, flexoki


# ------------------------------
# CLI
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--experiments",
        nargs='+',  # This allows multiple space-separated arguments like 'latent latentng'
        default=["latent", "latentng", "scalegmn", "lap", "neural_graphs", "naive"],
        # Removed 'type' as it creates a list which conflicts with 'choices' validation.
        # Removed 'choices' from here; validation will be handled manually in main()
        help=(
            "Experiments to plot. Space- or comma-separated, allowed: "
            '["naive", "lap", "scalegmn", "neural_graphs", "latent", "latentng"]'
        ),
    )
    p.add_argument(
        "--perturbation",
        type=float,
        required=True,
        help="Perturbation value used when generating matrices (e.g., 0.0 or 0.005).",
    )
    p.add_argument(
        "--orbit_transformation",
        type=str,
        choices=["PD", "P", "D"],
        required=True,
        help="Orbit/group transformation used when generating matrices.",
    )
    p.add_argument(
        "--matrices_dir",
        type=str,
        default="analysis/resources/interpolation/matrices",
        help="Directory containing the saved loss matrices.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="analysis/resources/interpolation",
        help="Directory to save the output plot.",
    )
    p.add_argument(
        "--band",
        type=str,
        default="std",
        choices=["sem", "std", "none"],
        help="Uncertainty band: standard error (sem), standard deviation (std), or none.",
    )
    p.add_argument(
        "--show_legend",
        default=False,
        action="store_true",
        help="Disable legend in the plot.",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom plot title.",
    )
    return p.parse_args()


# ------------------------------
# Helpers
# ------------------------------
LabelMap: Dict[str, str] = {
    "naive": "Naive",
    "scalegmn": "Autoencoder (ScaleGMN)",
    "lap": "Linear Assignment (Re-Basin)",
    "latent": "Latent (ScaleGMN)",
    "latentng": "Latent (Neural Graphs)",
    "neural_graphs": "Autoencoder (Neural Graphs)",
}

# Prefer Flexoki; otherwise default color cycle.
MethodColors: Dict[str, Tuple[str, int]] = {
    "Naive": ("Red", 500),
    "Autoencoder (ScaleGMN)": ("Yellow", 500),
    "Linear Assignment (Re-Basin)": ("Cyan", 500),
    "Autoencoder (Neural Graphs)": ("Purple", 500),
    "Latent (ScaleGMN)": ("Orange", 500),
    "Latent (Neural Graphs)": ("Green", 500),
}

def _curve_stats(matrix: torch.Tensor):
    """
    matrix: [N, T] where T is number of interpolation samples.
    Returns mean, std, sem, N (all numpy except N).
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected matrix [N, T], got {tuple(matrix.shape)}")
    mean = matrix.mean(dim=0).detach().cpu().numpy()
    std  = matrix.std(dim=0, unbiased=False).detach().cpu().numpy()
    n    = max(int(matrix.size(0)), 1)
    sem  = std / np.sqrt(n)
    return mean, std, sem, n

def _resolve_color(label: str):
    spec = MethodColors.get(label)
    if spec is None:
        return None
    name, shade = spec
    return plt_utils.flexoki(name, shade)


# ------------------------------
# Plot
# ------------------------------
def plot_curves(
    matrices: List[Tuple[torch.Tensor, str]],
    save_path: Path,
    title: Optional[str],
    show_legend: bool,
    band: str,
):
    # X axis inferred from matrix width (T)
    if not matrices:
        raise ValueError("No matrices to plot.")

    T = int(matrices[0][0].shape[1])
    xs = np.linspace(0.0, 1.0, T)

    # Set color cycle if available
    plt_utils.set_flexoki_cycle()

    plt.figure(figsize=(7.0, 4.5), dpi=150)

    for mat, label in matrices:
        mean, std, sem, _ = _curve_stats(mat)
        color = _resolve_color(label)
        plt.plot(xs, mean, marker='.', linestyle='-', linewidth=3.0, label=label, color=color)

        if band != "none":
            spread = sem if band == "sem" else std
            lower, upper = mean - spread, mean + spread
            print(f"[info] Plotting {label}: mean={mean.mean():.4f}, std={std.mean():.4f}, sem={sem.mean():.4f}")
            plt.fill_between(xs, lower, upper, alpha=0.30, linewidth=0, color=color)

    
    plt.xlabel("Interpolation $t$", fontsize=28)
    plt.ylabel("MSE loss", fontsize=28) 
    if title:
        plt.title(title)
    if show_legend:
        plt.legend(frameon=False)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


# ------------------------------
# Main
# ------------------------------
def main():
    args = parse_args()

    # Define valid choices for manual validation
    VALID_EXPERIMENT_CHOICES = ["latent", "latentng", "scalegmn", "lap", "neural_graphs", "naive"]

    # Process args.experiments to handle both space- and comma-separated inputs
    processed_experiments = []
    for item in args.experiments:
        if ',' in item:
            # This handles input like: --experiments "latent,lap"
            for sub_item in [s.strip() for s in item.split(',')]:
                if sub_item and sub_item not in VALID_EXPERIMENT_CHOICES:
                    print(f"Error: Invalid experiment choice '{sub_item}'. Allowed: {VALID_EXPERIMENT_CHOICES}", file=sys.stderr)
                    sys.exit(1)
                elif sub_item: # Add if not empty
                    processed_experiments.append(sub_item)
        else:
            # This handles input like: --experiments latent lap
            if item and item not in VALID_EXPERIMENT_CHOICES:
                print(f"Error: Invalid experiment choice '{item}'. Allowed: {VALID_EXPERIMENT_CHOICES}", file=sys.stderr)
                sys.exit(1)
            elif item: # Add if not empty
                processed_experiments.append(item)
    
    # Remove duplicates and sort for consistent processing
    args.experiments = list(sorted(set(processed_experiments)))

    # Exit if no valid experiments were specified
    if not args.experiments:
        print("[error] No valid experiments specified. Nothing to plot.", file=sys.stderr)
        sys.exit(1)

    experiments = args.experiments # This is now the clean, validated list
    matrices_dir = Path(args.matrices_dir)
    out_dir = Path(args.output_dir)
    orbit = args.orbit_transformation
    pert = args.perturbation

    loaded: List[Tuple[torch.Tensor, str]] = []

    for exp in experiments:
        path = matrices_dir / f"loss_matrix-{exp}-{orbit}-numruns=10-perturbation={pert}.pt"
        if not path.exists():
            print(f"[warning] Matrix for experiment '{exp}' not found at '{path}'. Skipping.", file=sys.stderr)
            continue
        matrix = torch.load(path, map_location="cpu")
        label = LabelMap.get(exp, exp)
        loaded.append((matrix, label))
        print(f"[info] Loaded {label} from '{path.name}', shape={tuple(matrix.shape)}")

    if not loaded:
        print("[error] No matrices loaded. Nothing to plot.", file=sys.stderr)
        sys.exit(1)

    # Title
    title = args.title if args.title else None

    exp_names_for_filename = "-".join(sorted(experiments)) # Use the processed list
    out_name_suffix = f"{exp_names_for_filename}-{orbit}-perturbation={pert}"
    out_path = out_dir / f"{out_name_suffix}.pdf"
    show_legend = args.show_legend 

    plot_curves(
        matrices=loaded,
        save_path=out_path,
        title=title,
        show_legend=show_legend,
        band=args.band,
    )

    print(f"[ok] Saved plot → {out_path}")

if __name__ == "__main__":
    main()