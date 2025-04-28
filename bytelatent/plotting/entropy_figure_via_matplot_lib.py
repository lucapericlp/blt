import os
import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_entropies(patch_lengths: torch.Tensor, scores: torch.Tensor, chars: str, threshold: float):
    patch_lengths_np = patch_lengths.cpu().numpy().flatten()
    scores_np = scores.cpu().float().numpy().flatten()
    chars = chars.replace(" ", "_")
    tokens_np = np.array([char for char in "<"+chars])

    if len(scores_np) != len(tokens_np):
        raise ValueError("Length of scores and tokens tensors must be the same.")
    if patch_lengths_np.sum() != len(tokens_np):
        raise ValueError(f"Sum of patch_lengths ({patch_lengths_np.sum()}) "
                        f"does not match the length of tokens/scores ({len(tokens_np)}).")


    x_indices = np.arange(len(tokens_np))

    # Calculate cumulative sums of patch lengths for vertical line positions
    # These indicate the *end* index of each patch
    patch_boundaries = np.cumsum(patch_lengths_np)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(15, 5)) # Adjust figure size as needed

    # Plot the scores as a blue line with markers
    ax.plot(x_indices, scores_np, marker='.', linestyle='-', color='steelblue', label='Scores')

    # Plot the vertical dotted lines at the patch boundaries
    # We plot a line *after* each patch, so at index `boundary - 1 + 0.5`
    # We skip the last boundary as it's the end of the data
    for boundary in patch_boundaries[:-1]:
        ax.axvline(x=boundary, color='grey', linestyle='--', linewidth=1)

    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1)

    # Set x-axis ticks and labels
    ax.set_xticks(x_indices)
    ax.set_xticklabels(tokens_np, rotation=0, fontsize=8) # Rotate labels for better readability

    # Set labels for axes
    # Using the Y-axis label from the example image
    ax.set_ylabel("Entropy of Next Byte", fontsize=12)
    ax.set_xlabel("Tokens", fontsize=12)

    # Set y-axis limits (optional, but often good practice)
    ax.set_ylim(bottom=0) # Start y-axis at 0 like the example
    ax.set_xlim(left = x_indices[0]-1.0, right = x_indices[-1]+1.0) # Add padding to x-axis

    # Add grid lines (optional)
    # ax.grid(True, axis='y', linestyle=':', color='lightgrey')

    # Remove the top and right spines for cleaner look (optional)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust layout and display the plot
    plt.tight_layout()
    output_filename = "token_score_plot.png"
    fig.savefig(output_filename, dpi=300, bbox_inches='tight') # Save the figure
    print(f"Plot saved to {os.path.abspath(output_filename)}") # Print confirmation with full path

    # Close the plot figure to free memory (good practice)
    plt.close(fig)
