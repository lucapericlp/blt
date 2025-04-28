import os
import gradio as gr
import torch
import itertools # For color cycling
import tiktoken # For GPT-4 tokenizer
from transformers import AutoTokenizer, AutoModel # For Llama3 tokenizer

# Bytelatent imports (assuming they are in the python path)
from bytelatent.data.file_util import get_fs
from bytelatent.generate_patcher import patcher_nocache
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
from bytelatent.plotting.entropy_figure_via_matplot_lib import plot_entropies
from bytelatent.args import TrainArgs
from download_blt_weights import main as ensure_present

# --- Global Setup ---

# Define colors for patches/tokens
VIZ_COLORS = [
    "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c",
    "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a", "#ffff99", "#b15928"
] # Add more if you expect many segments

LLAMA3_MODEL_NAME = "meta-llama/Meta-Llama-3-8B" # Or choose another variant like Instruct

# --- Helper Functions ---

def create_bytelatent_highlight_data(tokenizer, patch_lengths_tensor, tokens_tensor, colors):
    """Generates data for gr.HighlightedText based on bytelatent patches."""
    # (Keep the function from the previous version - no changes needed)
    if patch_lengths_tensor is None or tokens_tensor is None or patch_lengths_tensor.numel() == 0:
        return None
    patch_lengths = patch_lengths_tensor.tolist()
    all_tokens = tokens_tensor.tolist()
    highlighted_data = []
    current_token_index = 0
    color_cycler = itertools.cycle(colors)
    for i, length in enumerate(patch_lengths):
        if length <= 0: continue
        patch_token_ids = all_tokens[current_token_index : current_token_index + length]
        if not patch_token_ids: continue
        try: patch_text = tokenizer.decode(patch_token_ids)
        except Exception as decode_err:
             print(f"Warning: Bytelatent patch decoding failed: {decode_err}")
             patch_text = f"[Decode Error: {len(patch_token_ids)} tokens]"
        patch_label = f"BL Patch {i+1}"
        highlighted_data.append((patch_text, patch_label))
        current_token_index += length
    if current_token_index != len(all_tokens):
        print(f"Warning: Bytelatent token mismatch. Consumed {current_token_index}, total {len(all_tokens)}")
        remaining_tokens = all_tokens[current_token_index:]
        if remaining_tokens:
             try: remaining_text = tokenizer.decode(remaining_tokens)
             except Exception: remaining_text = f"[Decode Error: {len(remaining_tokens)} remaining tokens]"
             highlighted_data.append((remaining_text, "BL Remainder"))
    return highlighted_data


def create_tiktoken_highlight_data(prompt, colors):
    """Generates data for gr.HighlightedText based on tiktoken (gpt-4) tokens."""
    # (Keep the function from the previous version - no changes needed)
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        tiktoken_ids = enc.encode(prompt)
        highlighted_data = []
        color_cycler = itertools.cycle(colors)
        for i, token_id in enumerate(tiktoken_ids):
            try: token_text = enc.decode([token_id])
            except UnicodeDecodeError:
                 try:
                     token_bytes = enc.decode_single_token_bytes(token_id)
                     token_text = f"[Bytes: {token_bytes.hex()}]"
                 except Exception: token_text = "[Decode Error]"
            except Exception as e:
                 print(f"Unexpected tiktoken decode error: {e}")
                 token_text = "[Decode Error]"
            token_label = f"GPT4 Tk {i+1}"
            highlighted_data.append((token_text, token_label))
        print(f"Tiktoken processing complete. Found {len(tiktoken_ids)} tokens.")
        return highlighted_data
    except ImportError:
         print("Error: tiktoken library not found. Please install it: pip install tiktoken")
         return [("tiktoken library not installed.", "Error")]
    except Exception as tiktoken_err:
        print(f"Error during tiktoken processing: {tiktoken_err}")
        return [(f"Error processing with tiktoken: {str(tiktoken_err)}", "Error")]


def create_llama3_highlight_data(prompt, colors, model_name=LLAMA3_MODEL_NAME):
    """Generates data for gr.HighlightedText based on Llama 3 tokenizer."""
    try:
        # Load Llama 3 tokenizer from Hugging Face Hub
        # This might download the tokenizer files on the first run
        # May require `huggingface-cli login` if model is private or gated
        print(f"Loading Llama 3 tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Llama 3 tokenizer loaded.")

        # Encode the prompt
        llama_token_ids = tokenizer.encode(prompt)

        highlighted_data = []
        color_cycler = itertools.cycle(colors)

        for i, token_id in enumerate(llama_token_ids):
            try:
                # Decode individual token. Llama/SentencePiece tokenizers usually handle this well.
                token_text = tokenizer.decode([token_id])
                # Special case: Handle potential leading space added by sentencepiece during decode
                # if token_text.startswith(' '): # Check if this improves visualization
                #      token_text = token_text[1:] # Remove leading space visual artifact? Test this.
            except Exception as e:
                 print(f"Unexpected Llama 3 decode error for token {token_id}: {e}")
                 token_text = "[Decode Error]"

            token_label = f"Llama3 Tk {i+1}" # Clearer label prefix
            highlighted_data.append((token_text, token_label))

        print(f"Llama 3 processing complete. Found {len(llama_token_ids)} tokens.")
        return highlighted_data

    except ImportError:
         print("Error: transformers or sentencepiece library not found. Please install them: pip install transformers sentencepiece")
         return [("transformers/sentencepiece library not installed.", "Error")]
    except OSError as e:
        # Handle errors like model not found, network issues, authentication needed
        print(f"Error loading Llama 3 tokenizer '{model_name}': {e}")
        if "authentication" in str(e).lower():
             return [(f"Authentication required for Llama 3 tokenizer '{model_name}'. Use `huggingface-cli login`.", "Error")]
        else:
            return [(f"Could not load Llama 3 tokenizer '{model_name}'. Check model name and network. Error: {e}", "Error")]
    except Exception as llama_err:
        print(f"Error during Llama 3 processing: {llama_err}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return [(f"Error processing with Llama 3: {str(llama_err)}", "Error")]


# --- Main Processing Function ---

def process_text(prompt: str, model_name: str = "blt-1b"):
    """
    Processes the input prompt using ByteLatent, Tiktoken, and Llama 3,
    returning visualizations and status.

    Args:
        prompt: The input text string from the Gradio interface.
        model_name: The name of the bytelatent model to use.

    Returns:
        A tuple containing:
            - Matplotlib Figure for the entropy plot (or None).
            - List of tuples for bytelatent gr.HighlightedText (or None).
            - List of tuples for tiktoken gr.HighlightedText (or None).
            - List of tuples for Llama 3 gr.HighlightedText (or None).
            - Status/Error message string.
    """
    fig = None
    bl_highlighted_data = None
    tk_highlighted_data = None
    llama_highlighted_data = None
    status_message = "Starting processing..."

    # --- 1. Tiktoken Processing (Independent) ---
    status_message += "\nProcessing with Tiktoken (gpt-4)..."
    tk_highlighted_data = create_tiktoken_highlight_data(prompt, VIZ_COLORS)
    if tk_highlighted_data and tk_highlighted_data[0][1] == "Error":
         status_message += f"\nTiktoken Error: {tk_highlighted_data[0][0]}"
    else:
         status_message += "\nTiktoken processing successful."

    # --- 2. Llama 3 Processing (Independent) ---
    status_message += "\nProcessing with Llama 3 tokenizer..."
    llama_highlighted_data = create_llama3_highlight_data(prompt, VIZ_COLORS)
    if llama_highlighted_data and llama_highlighted_data[0][1] == "Error":
         status_message += f"\nLlama 3 Error: {llama_highlighted_data[0][0]}"
    else:
         status_message += "\nLlama 3 processing successful."

    # --- 3. Bytelatent Processing ---
    try:
        status_message += f"\nLoading entropy model for '{model_name}'..."
        # (Bytelatent loading code remains the same as previous version)
        consolidated_path = os.path.join("hf-weights", model_name)
        train_args_path = os.path.join(consolidated_path, "params.json")
        if not os.path.exists(train_args_path): raise FileNotFoundError(f"Bytelatent training args not found at {train_args_path}.")
        fs = get_fs(train_args_path); train_args = TrainArgs.model_validate_json(fs.read_text(train_args_path))
        bl_tokenizer = train_args.data.tokenizer_args.build(); assert isinstance(bl_tokenizer, BltTokenizer)
        patcher_args = train_args.data.patcher_args.model_copy(deep=True); patcher_args.realtime_patching = True
        device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Using Bytelatent device: {device}")
        patcher_args.patching_device = device; patcher_args.device = device
        entropy_model_dir = os.path.join(consolidated_path, "entropy_model")
        if not os.path.exists(entropy_model_dir): raise FileNotFoundError(f"Bytelatent entropy model directory not found at {entropy_model_dir}.")
        patcher_args.entropy_model_checkpoint_dir = entropy_model_dir; bl_patcher = patcher_args.build()
        status_message += "\nBytelatent model loaded."

        # --- Processing ---
        status_message += "\nRunning Bytelatent patching..."
        print(f"Processing prompt with Bytelatent: '{prompt}'")
        # Limit prompt length for bytelatent if necessary
        prompt_bytes = prompt.encode('utf-8')
        if len(prompt_bytes) > 512:
             print(f"Warning: Prompt exceeds 512 bytes ({len(prompt_bytes)}). Truncating for Bytelatent.")
             prompt_bl = prompt_bytes[:512].decode('utf-8', errors='ignore')
             status_message += "\nWarning: Prompt truncated to 512 bytes for Bytelatent."
        else:
             prompt_bl = prompt

        results = patcher_nocache([prompt_bl], tokenizer=bl_tokenizer, patcher=bl_patcher)

        if not results:
            print("Bytelatent processing returned no results.")
            status_message += "\nBytelatent Warning: Processing completed, but no results were generated."
        else:
            batch_patch_lengths, batch_scores, batch_tokens = results
            patch_lengths, scores, tokens = batch_patch_lengths[0], batch_scores[0], batch_tokens[0]
            # --- Visualization Data Generation ---
            try: decoded_output_for_plot = bl_tokenizer.decode(tokens.tolist())
            except Exception as decode_err:
                 print(f"Warning: Error decoding full sequence for plot: {decode_err}")
                 decoded_output_for_plot = prompt_bl # Use truncated prompt for plot if decode fails
            fig = plot_entropies(patch_lengths, scores, decoded_output_for_plot, threshold=bl_patcher.threshold)
            bl_highlighted_data = create_bytelatent_highlight_data(bl_tokenizer, patch_lengths, tokens, VIZ_COLORS)
            status_message += "\nBytelatent processing and visualization successful."
            print("Bytelatent processing and decoding complete.")

    except FileNotFoundError as e:
        print(f"Bytelatent Error: {e}")
        status_message += f"\nBytelatent FileNotFoundError: {str(e)}"
    except Exception as e:
        print(f"An unexpected Bytelatent error occurred: {e}")
        import traceback
        traceback.print_exc()
        status_message += f"\nBytelatent Unexpected Error: {str(e)}"

    # Return all generated data and the final status message
    return fig, bl_highlighted_data, tk_highlighted_data, llama_highlighted_data, status_message


# --- Gradio Interface ---

# Create color maps for HighlightedText dynamically
MAX_EXPECTED_SEGMENTS = 1000 # Increase max expected segments further
common_error_map = {"Error": "#FF0000"} # Red for errors

bytelatent_color_map = {f"BL Patch {i+1}": color for i, color in zip(range(MAX_EXPECTED_SEGMENTS), itertools.cycle(VIZ_COLORS))}
bytelatent_color_map["BL Remainder"] = "#808080"; bytelatent_color_map.update(common_error_map)

tiktoken_color_map = {f"GPT4 Tk {i+1}": color for i, color in zip(range(MAX_EXPECTED_SEGMENTS), itertools.cycle(VIZ_COLORS))}
tiktoken_color_map.update(common_error_map)

llama3_color_map = {f"Llama3 Tk {i+1}": color for i, color in zip(range(MAX_EXPECTED_SEGMENTS), itertools.cycle(VIZ_COLORS))}
llama3_color_map.update(common_error_map)


with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown("# BLT's Entropy Patcher Visualisation") # Updated Title
    gr.Markdown(
        "Enter text to visualize its segmentation according to different tokenizers:\n"
        "1.  **BLT:** Entropy plot and text segmented by dynamic patches (Input limited to 512 bytes).\n"
        "2.  **Tiktoken (GPT-4):** Text segmented by `cl100k_base` tokens.\n"
        "3.  **Llama 3:** Text segmented by the `meta-llama/Meta-Llama-3-8B` tokenizer."
    )

    with gr.Row():
         with gr.Column(scale=1): # Input Column
            prompt_input = gr.Textbox(
                label="Input Prompt",
                value="Daenerys Targaryen is in Game of Thrones, a fantasy epic by George R.R. Martin.",
                placeholder="Enter text here...",
                max_length=2048, # Allow even longer input, Bytelatent will truncate
                lines=5,
                info="Processing is limited to the first 512 bytes of the input."
            )
            submit_button = gr.Button("Generate Visualizations", variant="primary")
            status_output = gr.Textbox(label="Processing Status", interactive=False, lines=5)

         with gr.Column(scale=2): # Output Column
             gr.Markdown("### BLT's Entropy Patcher Output (`100m`)")
             highlighted_output_bl = gr.HighlightedText(
                 label="Bytelatent Patched Text",
                 color_map=bytelatent_color_map,
                 show_legend=False, # Legend can get very long, disable for compactness
                 show_inline_category=False,
             )
             plot_output = gr.Plot(label="Bytelatent Entropy vs. Token Index")

             gr.Markdown("### Tiktoken Output (`cl100k_base` for GPT-4)")
             highlighted_output_tk = gr.HighlightedText(
                 label="Tiktoken Segmented Text",
                 color_map=tiktoken_color_map,
                 show_legend=False,
                 show_inline_category=False,
             )

             gr.Markdown(f"### Llama 3 Output (`{LLAMA3_MODEL_NAME}`)")
             highlighted_output_llama = gr.HighlightedText(
                 label="Llama 3 Segmented Text",
                 color_map=llama3_color_map,
                 show_legend=False,
                 show_inline_category=False,
             )

    # Define the action for the button click
    submit_button.click(
        fn=process_text,
        inputs=prompt_input,
        # Ensure order matches the 5 return values of process_text
        outputs=[
            plot_output,
            highlighted_output_bl,
            highlighted_output_tk,
            highlighted_output_llama,
            status_output
            ]
    )

# --- Launch the Gradio App ---
if __name__ == "__main__":
    print("Please ensure 'tiktoken', 'transformers', and 'sentencepiece' are installed (`pip install tiktoken transformers sentencepiece`)")
    print(f"Attempting to use Llama 3 Tokenizer: {LLAMA3_MODEL_NAME}. Ensure you have access (e.g., via `huggingface-cli login` if needed).")
    ensure_present(["blt-1b"]) # Ensure bytelatent model is present
    iface.launch()
