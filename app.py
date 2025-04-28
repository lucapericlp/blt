import os
import gradio as gr
import torch
import itertools # For color cycling
import tiktoken # For GPT-4 tokenizer
from transformers import AutoTokenizer # For Llama3 tokenizer - AutoModel usually not needed just for tokenizer

# Bytelatent imports (assuming they are in the python path)
try:
    from bytelatent.data.file_util import get_fs
    from bytelatent.generate_patcher import patcher_nocache
    from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
    from bytelatent.plotting.entropy_figure_via_matplot_lib import plot_entropies
    from bytelatent.args import TrainArgs
    from download_blt_weights import main as ensure_present
    BLT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Bytelatent libraries not found. Bytelatent functionality will be disabled. Error: {e}")
    BLT_AVAILABLE = False
    # Define dummy classes/functions if BLT is not available to avoid NameErrors later
    class BltTokenizer: pass
    class TrainArgs: pass
    def patcher_nocache(*args, **kwargs): return None
    def plot_entropies(*args, **kwargs): return None
    def ensure_present(*args, **kwargs): pass


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
    if not BLT_AVAILABLE:
        return [("Bytelatent library not available.", "Error")]
    if patch_lengths_tensor is None or tokens_tensor is None or patch_lengths_tensor.numel() == 0:
        return None
    patch_lengths = patch_lengths_tensor.tolist()
    all_tokens = tokens_tensor.tolist()
    highlighted_data = []
    current_token_index = 0
    patch_count = 0 # Initialize patch count
    # color_cycler = itertools.cycle(colors) # Moved inside loop if needed per-patch
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
        patch_count += 1 # Increment count for each valid patch added
        current_token_index += length

    # Handle remainder separately, don't count it as a 'patch'
    if current_token_index != len(all_tokens):
        print(f"Warning: Bytelatent token mismatch. Consumed {current_token_index}, total {len(all_tokens)}")
        remaining_tokens = all_tokens[current_token_index:]
        if remaining_tokens:
            try: remaining_text = tokenizer.decode(remaining_tokens)
            except Exception: remaining_text = f"[Decode Error: {len(remaining_tokens)} remaining tokens]"
            highlighted_data.append((remaining_text, "BL Remainder"))

    # Return both highlighted data and the calculated patch count
    return highlighted_data, patch_count


def create_tiktoken_highlight_data(prompt, colors):
    """Generates data for gr.HighlightedText based on tiktoken (gpt-4) tokens."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        tiktoken_ids = enc.encode(prompt)
        highlighted_data = []
        # color_cycler = itertools.cycle(colors) # Moved inside loop if needed per-token
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
        token_count = len(tiktoken_ids)
        print(f"Tiktoken processing complete. Found {token_count} tokens.")
        return highlighted_data, token_count
    except ImportError:
        print("Error: tiktoken library not found. Please install it: pip install tiktoken")
        return [("tiktoken library not installed.", "Error")], 0
    except Exception as tiktoken_err:
        print(f"Error during tiktoken processing: {tiktoken_err}")
        return [(f"Error processing with tiktoken: {str(tiktoken_err)}", "Error")], 0


def create_llama3_highlight_data(prompt, colors, model_name=LLAMA3_MODEL_NAME):
    """Generates data for gr.HighlightedText based on Llama 3 tokenizer."""
    try:
        # Load Llama 3 tokenizer from Hugging Face Hub
        print(f"Loading Llama 3 tokenizer: {model_name}")
        # Use trust_remote_code=True if required by the specific model revision
        tokenizer = AutoTokenizer.from_pretrained(model_name) #, trust_remote_code=True)
        print("Llama 3 tokenizer loaded.")

        # Encode the prompt
        llama_token_ids = tokenizer.encode(prompt)

        highlighted_data = []
        # color_cycler = itertools.cycle(colors) # Moved inside loop if needed per-token

        for i, token_id in enumerate(llama_token_ids):
            try:
                # Decode individual token.
                token_text = tokenizer.decode([token_id])
            except Exception as e:
                print(f"Unexpected Llama 3 decode error for token {token_id}: {e}")
                token_text = "[Decode Error]"

            token_label = f"Llama3 Tk {i+1}" # Clearer label prefix
            highlighted_data.append((token_text, token_label))

        token_count = len(llama_token_ids)
        print(f"Llama 3 processing complete. Found {token_count} tokens.")
        return highlighted_data, token_count

    except ImportError:
        print("Error: transformers or sentencepiece library not found. Please install them: pip install transformers sentencepiece")
        return [("transformers/sentencepiece library not installed.", "Error")], 0
    except OSError as e:
        # Handle errors like model not found, network issues, authentication needed
        print(f"Error loading Llama 3 tokenizer '{model_name}': {e}")
        error_msg = f"Could not load Llama 3 tokenizer '{model_name}'. Check model name and network."
        if "authentication" in str(e).lower():
             error_msg = f"Authentication required for Llama 3 tokenizer '{model_name}'. Use `huggingface-cli login`."
        return [(f"{error_msg} Error: {e}", "Error")], 0
    except Exception as llama_err:
        print(f"Error during Llama 3 processing: {llama_err}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return [(f"Error processing with Llama 3: {str(llama_err)}", "Error")], 0


# --- Main Processing Function ---

def process_text(prompt: str, model_name: str = "blt-1b"):
    """
    Processes the input prompt using ByteLatent, Tiktoken, and Llama 3,
    returning visualizations, counts, and status.

    Args:
        prompt: The input text string from the Gradio interface.
        model_name: The name of the bytelatent model to use.

    Returns:
        A tuple containing:
            - Matplotlib Figure for the entropy plot (or None).
            - List of tuples for bytelatent gr.HighlightedText (or None).
            - Integer count of bytelatent patches.
            - List of tuples for tiktoken gr.HighlightedText (or None).
            - Integer count of tiktoken tokens.
            - List of tuples for Llama 3 gr.HighlightedText (or None).
            - Integer count of Llama 3 tokens.
            - Status/Error message string.
    """
    fig = None
    bl_highlighted_data = None
    tk_highlighted_data = None
    llama_highlighted_data = None
    bl_count = 0
    tk_count = 0
    llama_count = 0
    status_message = "Starting processing..."

    # --- 1. Tiktoken Processing (Independent) ---
    status_message += "\nProcessing with Tiktoken (gpt-4)..."
    tk_highlighted_data, tk_count_calc = create_tiktoken_highlight_data(prompt, VIZ_COLORS)
    if tk_highlighted_data and tk_highlighted_data[0][1] == "Error":
        status_message += f"\nTiktoken Error: {tk_highlighted_data[0][0]}"
        tk_count = 0 # Ensure count is 0 on error
    else:
        tk_count = tk_count_calc # Assign calculated count
        status_message += f"\nTiktoken processing successful ({tk_count} tokens)."

    # --- 2. Llama 3 Processing (Independent) ---
    status_message += "\nProcessing with Llama 3 tokenizer..."
    llama_highlighted_data, llama_count_calc = create_llama3_highlight_data(prompt, VIZ_COLORS)
    if llama_highlighted_data and llama_highlighted_data[0][1] == "Error":
        status_message += f"\nLlama 3 Error: {llama_highlighted_data[0][0]}"
        llama_count = 0 # Ensure count is 0 on error
    else:
        llama_count = llama_count_calc # Assign calculated count
        status_message += f"\nLlama 3 processing successful ({llama_count} tokens)."

    # --- 3. Bytelatent Processing ---
    if BLT_AVAILABLE:
        try:
            status_message += f"\nLoading Bytelatent entropy model for '{model_name}'..."
            # (Bytelatent loading code remains the same)
            consolidated_path = os.path.join("hf-weights", model_name)
            train_args_path = os.path.join(consolidated_path, "params.json")
            if not os.path.exists(train_args_path): raise FileNotFoundError(f"BLT training args not found at {train_args_path}.")
            fs = get_fs(train_args_path); train_args = TrainArgs.model_validate_json(fs.read_text(train_args_path))
            bl_tokenizer = train_args.data.tokenizer_args.build(); assert isinstance(bl_tokenizer, BltTokenizer)
            patcher_args = train_args.data.patcher_args.model_copy(deep=True); patcher_args.realtime_patching = True
            device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Using BLT device: {device}")
            patcher_args.patching_device = device; patcher_args.device = device
            entropy_model_dir = os.path.join(consolidated_path, "entropy_model")
            if not os.path.exists(entropy_model_dir): raise FileNotFoundError(f"Entropy model directory not found at {entropy_model_dir}.")
            patcher_args.entropy_model_checkpoint_dir = entropy_model_dir; bl_patcher = patcher_args.build()
            status_message += "\nBytelatent entropy model loaded."

            # --- Processing ---
            status_message += "\nRunning Bytelatent entropy model patching..."
            print(f"Processing prompt with entropy model: '{prompt}'")
            prompt_bytes = prompt.encode('utf-8')
            max_bytes = 512 # Define max bytes
            if len(prompt_bytes) > max_bytes:
                print(f"Warning: Prompt exceeds {max_bytes} bytes ({len(prompt_bytes)}). Truncating for entropy model.")
                # Find the byte position that corresponds to the last full character within the limit
                # This avoids splitting a multi-byte character
                try:
                    last_char_pos = prompt_bytes[:max_bytes].rfind(b' ') # Simple whitespace split point find, might not be perfect
                    if last_char_pos == -1: # If no space, truncate hard (less ideal)
                         prompt_bl = prompt_bytes[:max_bytes].decode('utf-8', errors='ignore')
                    else:
                         prompt_bl = prompt_bytes[:last_char_pos].decode('utf-8', errors='ignore')

                except Exception: # Fallback to simple truncation on decode errors
                     prompt_bl = prompt_bytes[:max_bytes].decode('utf-8', errors='ignore')

                status_message += f"\nWarning: Prompt truncated to approx {len(prompt_bl.encode('utf-8'))} bytes for Bytelatent entropy model."
            else:
                prompt_bl = prompt

            results = patcher_nocache([prompt_bl], tokenizer=bl_tokenizer, patcher=bl_patcher)

            if not results:
                print("Bytelatent entropy processing returned no results.")
                status_message += "\nBytelatent entropy model warning: Processing completed, but no results were generated."
                bl_highlighted_data = [("No patches generated.", "Info")]
                bl_count = 0
            else:
                batch_patch_lengths, batch_scores, batch_tokens = results
                patch_lengths, scores, tokens = batch_patch_lengths[0], batch_scores[0], batch_tokens[0]
                # --- Visualization Data Generation ---
                try: decoded_output_for_plot = bl_tokenizer.decode(tokens.tolist())
                except Exception as decode_err:
                    print(f"Warning: Error decoding full sequence for plot: {decode_err}")
                    decoded_output_for_plot = prompt_bl # Use truncated prompt for plot if decode fails

                fig = plot_entropies(patch_lengths, scores, decoded_output_for_plot, threshold=bl_patcher.threshold)
                bl_highlighted_data, bl_count_calc = create_bytelatent_highlight_data(bl_tokenizer, patch_lengths, tokens, VIZ_COLORS)
                bl_count = bl_count_calc # Assign calculated count

                status_message += f"\nBytelatent entropy model processing and visualization successful ({bl_count} patches)."
                print("Bytelatent Entropy model processing and decoding complete.")

        except FileNotFoundError as e:
            print(f"Bytelatent Error: {e}")
            status_message += f"\nBytelatent FileNotFoundError: {str(e)}"
            bl_highlighted_data = [(f"Bytelatent Error: {e}", "Error")]
            bl_count = 0
        except Exception as e:
            print(f"An unexpected Bytelatent error occurred: {e}")
            import traceback
            traceback.print_exc()
            status_message += f"\nBytelatent Unexpected Error: {str(e)}"
            bl_highlighted_data = [(f"Bytelatent Error: {e}", "Error")]
            bl_count = 0
    else:
         status_message += "\nBytelatent processing skipped (library not found)."
         bl_highlighted_data = [("Bytelatent library not available.", "Error")]
         bl_count = 0
         fig = None # Ensure fig is None if BLT is skipped

    # Return all generated data and the final status message
    return fig, bl_highlighted_data, bl_count, tk_highlighted_data, tk_count, llama_highlighted_data, llama_count, status_message


# --- Gradio Interface ---

# Create color maps for HighlightedText dynamically
MAX_EXPECTED_SEGMENTS = 2000 # Increased max segments further just in case
common_error_map = {"Error": "#FF0000", "Info": "#808080"} # Red for errors, Gray for info

bytelatent_color_map = {f"BL Patch {i+1}": color for i, color in zip(range(MAX_EXPECTED_SEGMENTS), itertools.cycle(VIZ_COLORS))}
bytelatent_color_map["BL Remainder"] = "#AAAAAA"; bytelatent_color_map.update(common_error_map)

tiktoken_color_map = {f"GPT4 Tk {i+1}": color for i, color in zip(range(MAX_EXPECTED_SEGMENTS), itertools.cycle(VIZ_COLORS))}
tiktoken_color_map.update(common_error_map)

llama3_color_map = {f"Llama3 Tk {i+1}": color for i, color in zip(range(MAX_EXPECTED_SEGMENTS), itertools.cycle(VIZ_COLORS))}
llama3_color_map.update(common_error_map)


with gr.Blocks(theme=gr.themes.Origin()) as iface:
    gr.Markdown("# BLT's Entropy-based Patcher vs. Tokenizer Visualisation")
    gr.Markdown(
        "Enter text to visualize its segmentation according to different methods:\n"
        "1.  **Byte Latent Transformer (BLT):** Entropy-based patching plot and patched text (_for this space ONLY_ - limited to ~512 bytes).\n"
        "2.  **Tiktoken (GPT-4):** Text segmented by `cl100k_base` tokens.\n"
        f"3.  **Llama 3:** Text segmented by the `{LLAMA3_MODEL_NAME}` tokenizer."
    )

    with gr.Row():
        with gr.Column(scale=1): # Input Column
            prompt_input = gr.Textbox(
                label="Input Prompt",
                value="Daenerys Targaryen is in Game of Thrones, a fantasy epic by George R.R. Martin.",
                placeholder="Enter text here...",
                max_length=512, # Allow even longer input, Bytelatent will truncate
                lines=5,
                info="For this space ONLY, processing is limited to ~512 bytes."
            )
            submit_button = gr.Button("Generate Visualizations", variant="primary")
            status_output = gr.Textbox(label="Processing Status", interactive=False, lines=7) # Increased lines slightly

        with gr.Column(scale=2): # Output Column

            # --- Bytelatent Output Area ---
            with gr.Row(equal_height=False): # Use Row to place title and count together
                 gr.Markdown("### BLT Entropy Patcher Output (`blt_main_entropy_100m_512w`)")

            bl_count_output = gr.Number(label="Patch Count", value=0, interactive=False, scale=1, step=1) # Added Number output
            highlighted_output_bl = gr.HighlightedText(
                label="BLT's Entropy-based Patches",
                color_map=bytelatent_color_map,
                show_legend=False, # Legend can get very long
                # show_label=False, # Hide the HighlightedText label as we have the markdown title
                show_inline_category=False,
                # container=False, # Reduces vertical space slightly
            )
            plot_output = gr.Plot(label="Entropy vs. Token Index", show_label=True)

            # --- Tiktoken Output Area ---
            with gr.Row(equal_height=False):
                gr.Markdown("### Tiktoken Output (`cl100k_base`)")

            tk_count_output = gr.Number(label="Token Count", value=0, interactive=False, scale=1, step=1) # Added Number output
            highlighted_output_tk = gr.HighlightedText(
                label="Tiktoken Segmented Text",
                color_map=tiktoken_color_map,
                show_legend=False,
                show_inline_category=False,
                # show_label=False,
                # container=False,
            )

            # --- Llama 3 Output Area ---
            with gr.Row(equal_height=False):
                gr.Markdown(f"### Llama 3 Output (`{LLAMA3_MODEL_NAME}`)")

            llama_count_output = gr.Number(label="Token Count", value=0, interactive=False, scale=1, step=1) # Added Number output
            highlighted_output_llama = gr.HighlightedText(
                label="Llama 3 Segmented Text",
                color_map=llama3_color_map,
                show_legend=False,
                show_inline_category=False,
                # show_label=False,
                # container=False,
            )

    # Define the action for the button click
    submit_button.click(
        fn=process_text,
        inputs=prompt_input,
        # Ensure order matches the 8 return values of process_text
        outputs=[
            plot_output,             # fig
            highlighted_output_bl,   # bl_highlighted_data
            bl_count_output,         # bl_count          <-- New
            highlighted_output_tk,   # tk_highlighted_data
            tk_count_output,         # tk_count          <-- New
            highlighted_output_llama,# llama_highlighted_data
            llama_count_output,      # llama_count       <-- New
            status_output            # status_message
            ]
    )

# --- Launch the Gradio App ---
if __name__ == "__main__":
    print("Checking required libraries...")
    try:
        import tiktoken
        print("- tiktoken found.")
    except ImportError:
        print("WARNING: 'tiktoken' not found. GPT-4 visualization will fail. Install with: pip install tiktoken")
    try:
        import transformers
        import sentencepiece
        print("- transformers found.")
        print("- sentencepiece found.")
    except ImportError:
         print("WARNING: 'transformers' or 'sentencepiece' not found. Llama 3 visualization will fail. Install with: pip install transformers sentencepiece")

    if BLT_AVAILABLE:
        print("- Bytelatent libraries found.")
        # Ensure bytelatent model is present only if library is available
        try:
            print("Ensuring Bytelatent model 'blt-1b' weights are present...")
            ensure_present(["blt-1b"])
            print("Bytelatent model check complete.")
        except Exception as blt_dl_err:
            print(f"WARNING: Failed to ensure Bytelatent model presence: {blt_dl_err}")
    else:
        print("INFO: Bytelatent libraries not found, skipping related functionality.")

    print(f"Attempting to use Llama 3 Tokenizer: {LLAMA3_MODEL_NAME}. Ensure you have access (e.g., via `huggingface-cli login` if needed).")
    print("Launching Gradio interface...")
    iface.launch()
