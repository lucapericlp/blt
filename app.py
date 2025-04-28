import os
import gradio as gr
import torch
import itertools # Import itertools for color cycling

from bytelatent.data.file_util import get_fs
from bytelatent.generate_patcher import patcher_nocache
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
from bytelatent.plotting.entropy_figure_via_matplot_lib import plot_entropies
from bytelatent.args import TrainArgs
from download_blt_weights import main as ensure_present

# --- Global Setup (Consider loading models outside if necessary) ---
# Kept inside the function for simplicity as before.

# Define colors for patches (similar to the image style)
# Using colors from a qualitative colormap (e.g., Colorbrewer Set3 or Paired)
PATCH_COLORS = [
    "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c",
    "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a", "#ffff99", "#b15928"
] # Add more if you expect many patches


def create_highlighted_text_data(tokenizer, patch_lengths_tensor, tokens_tensor, colors):
    """
    Generates the data structure needed for gr.HighlightedText based on patches.

    Args:
        tokenizer: The BltTokenizer instance.
        patch_lengths_tensor: Tensor containing the length of each patch (in tokens).
        tokens_tensor: Tensor containing the token IDs for the entire sequence.
        colors: A list of color hex codes to cycle through.

    Returns:
        A list of tuples for gr.HighlightedText, e.g., [(text, label), ...].
        Returns None if input tensors are invalid.
    """
    if patch_lengths_tensor is None or tokens_tensor is None or patch_lengths_tensor.numel() == 0:
        return None

    patch_lengths = patch_lengths_tensor.tolist()
    all_tokens = tokens_tensor.tolist()
    highlighted_data = []
    current_token_index = 0
    color_cycler = itertools.cycle(colors) # Use itertools to cycle through colors

    for i, length in enumerate(patch_lengths):
        if length <= 0: # Skip empty patches if they somehow occur
             continue
        patch_token_ids = all_tokens[current_token_index : current_token_index + length]
        if not patch_token_ids: # Should not happen if length > 0, but good practice
            continue

        patch_text = tokenizer.decode(patch_token_ids)
        patch_label = f"Patch {i+1}" # Unique label for each patch
        patch_color = next(color_cycler) # Get the next color

        # Add to highlighted_data: (text, label_for_coloring)
        highlighted_data.append((patch_text, patch_label))
        current_token_index += length

    # Check if all tokens were consumed (optional sanity check)
    if current_token_index != len(all_tokens):
        print(f"Warning: Token mismatch. Consumed {current_token_index}, total {len(all_tokens)}")
        # Decode any remaining tokens if necessary, though this indicates a logic issue
        remaining_tokens = all_tokens[current_token_index:]
        if remaining_tokens:
             remaining_text = tokenizer.decode(remaining_tokens)
             highlighted_data.append((remaining_text, "Remainder")) # Assign a generic label

    return highlighted_data


def process_text(prompt: str, model_name: str = "blt-1b"):
    """
    Processes the input prompt using the ByteLatent model and returns
    an entropy plot and color-coded text data.

    Args:
        prompt: The input text string from the Gradio interface.
        model_name: The name of the model to use.

    Returns:
        A tuple containing:
            - Matplotlib Figure for the entropy plot (or None on error).
            - List of tuples for gr.HighlightedText (or None on error/no results).
            - Error message string (or None if successful).
    """
    try:
        # --- Model and Tokenizer Loading ---
        consolidated_path = os.path.join("hf-weights", model_name)
        train_args_path = os.path.join(consolidated_path, "params.json")

        if not os.path.exists(train_args_path):
             raise FileNotFoundError(f"Training args not found at {train_args_path}. "
                                     f"Ensure model '{model_name}' is downloaded/available.")

        fs = get_fs(train_args_path)
        train_args = TrainArgs.model_validate_json(fs.read_text(train_args_path))

        tokenizer = train_args.data.tokenizer_args.build()
        assert isinstance(tokenizer, BltTokenizer)

        patcher_args = train_args.data.patcher_args.model_copy(deep=True)
        patcher_args.realtime_patching = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        patcher_args.patching_device = device
        patcher_args.device = device

        print("Loading entropy model and patcher...")
        entropy_model_dir = os.path.join(consolidated_path, "entropy_model")
        if not os.path.exists(entropy_model_dir):
             raise FileNotFoundError(f"Entropy model directory not found at {entropy_model_dir}.")

        patcher_args.entropy_model_checkpoint_dir = entropy_model_dir
        patcher = patcher_args.build()
        # --- End Loading ---

        # --- Processing ---
        prompts = [prompt]
        print(f"Processing prompt: '{prompt}'")
        results = patcher_nocache(
            prompts, tokenizer=tokenizer, patcher=patcher
        )

        if not results:
            print("Processing returned no results.")
            return None, None, "Processing completed, but no results were generated."

        batch_patch_lengths, batch_scores, batch_tokens = results

        # Process the first (and only) result in the batch
        patch_lengths = batch_patch_lengths[0]
        scores = batch_scores[0]
        tokens = batch_tokens[0]

        # Decode the full output once for the plot labels (if needed by plot_entropies)
        # Note: BltTokenizer might decode directly to bytes, then utf-8. Ensure it handles errors.
        try:
            # Using the raw tokens tensor for decoding consistency
            decoded_output_for_plot = tokenizer.decode(tokens.tolist())
        except Exception as decode_err:
            print(f"Warning: Error decoding full sequence for plot: {decode_err}")
            # Fallback: attempt to decode the original prompt if possible, or use generic labels
            decoded_output_for_plot = prompt # Use original prompt as fallback

        # Generate the plot
        fig = plot_entropies(
            patch_lengths,
            scores,
            decoded_output_for_plot, # Pass the decoded string for plot labels
            threshold=patcher.threshold
        )

        # Generate data for HighlightedText
        highlighted_data = create_highlighted_text_data(
             tokenizer, patch_lengths, tokens, PATCH_COLORS
        )

        print("Processing and visualization data generation complete.")
        # --- End Processing ---

        return fig, highlighted_data, None # Return plot, highlighted text data, no error

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None, f"Error: {str(e)}" # Return None for plot/text, error message
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None, None, f"An unexpected error occurred: {e}" # Return None for plot/text, error message

# --- Gradio Interface ---

# Create the color map for HighlightedText dynamically
# Generate enough patch labels and map them to the cycled colors
MAX_EXPECTED_PATCHES = 50 # Estimate a reasonable maximum
color_map = {
    f"Patch {i+1}": color
    for i, color in zip(range(MAX_EXPECTED_PATCHES), itertools.cycle(PATCH_COLORS))
}
# Add a color for the potential 'Remainder' label from create_highlighted_text_data
color_map["Remainder"] = "#808080" # Grey for any leftovers

with gr.Blocks() as iface:
    gr.Markdown("# ByteLatent Entropy Visualizer") # Title
    gr.Markdown(
        "Process any prompt (limited to 512 bytes) with the 100M entropy patcher model "
        "and visualize the token entropies plot and color-coded patches below.<br><br>" # Updated description
        "NOTE: this implementation differs slightly by excluding local attention so we limit "
        "the characters limit to 512 to avoid any deviation.",
        line_breaks=True
    )

    with gr.Column():
        prompt_input = gr.Textbox(
            label="Input Prompt",
            value="Daenerys Targaryen is in Game of Thrones, a fantasy epic by George R.R. Martin.",
            placeholder="Enter text here...",
            max_length=512,
            lines=3
        )
        submit_button = gr.Button("Generate Visualization") # Update button text

        # Output for error messages or status
        status_output = gr.Textbox(label="Status", interactive=False)

        # Output component for the color-coded text
        highlighted_output = gr.HighlightedText(
            label="Patched Text Visualization",
            color_map=color_map,
            show_legend=False # Show the patch labels and colors
        )

        # Output component for the plot
        plot_output = gr.Plot(label="Entropy vs. Token Index (with Patch Threshold)")

    # Define the action for the button click
    submit_button.click(
        fn=process_text,
        inputs=prompt_input,
        outputs=[plot_output, highlighted_output, status_output] # Order matters!
    )

# --- Launch the Gradio App ---
if __name__ == "__main__":
    ensure_present(["blt-1b"]) # Ensure model is present before launching
    iface.launch()
