# --- START OF FILE analysis/plot_targeted_interpolation.py ---

import argparse
import pathlib
import torch
import sys

# Add the project root to the Python path to allow importing from 'utils'
# You might need to adjust this path depending on your project structure.
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from analysis.utils.utils import plot_interpolation_curves

def get_args():
    """Parses command-line arguments for the targeted plotting script."""
    p = argparse.ArgumentParser(description="Plot a targeted comparison of interpolation methods.")
    
    p.add_argument("--naive_matrix_path", type=str, required=True,
                   help="Path to the loss matrix from the 'naive' interpolation.")
                   
    p.add_argument("--scalegmn_rebased_path", type=str, required=True,
                   help="Path to the loss matrix from the ScaleGMN 'rebased' interpolation.")
                   
    p.add_argument("--ng_rebased_path", type=str, required=True,
                   help="Path to the loss matrix from the Neural Graphs 'rebased' interpolation.")

    p.add_argument("--out_plot_path", type=str, required=True,
                   help="Full path where the output plot PNG will be saved.")
                   
    return p.parse_args()

def main():
    """Loads the specified loss matrices and generates a comparison plot."""
    args = get_args()

    print("Loading loss matrices for targeted comparison...")

    # Load the three required matrices
    try:
        naive_matrix = torch.load(args.naive_matrix_path)
        scalegmn_matrix = torch.load(args.scalegmn_rebased_path)
        ng_matrix = torch.load(args.ng_rebased_path)
        print("Matrices loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Could not find one of the matrix files. {e}")
        sys.exit(1)

    # Define the list of curves to plot, each as a (matrix, label) tuple
    curves_to_plot = [
        (naive_matrix, "Naive Interpolation"),
        (scalegmn_matrix, "ScaleGMN Re-based"),
        (ng_matrix, "Neural Graphs Re-based")
    ]
    
    # Ensure the output directory exists
    output_dir = pathlib.Path(args.out_plot_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating plot at: {args.out_plot_path}")
    
    # Call the plotting utility function
    # Note: I'm assuming your plot_interpolation_curves can accept a `title` argument.
    # If not, you may need to add it or it will use a default title.
    plot_interpolation_curves(
        curves_to_plot,
        save_path=args.out_plot_path,
        with_legend=True,
    )
    
    print(f"Plot saved successfully to {args.out_plot_path}")

if __name__ == "__main__":
    main()