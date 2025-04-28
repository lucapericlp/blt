import os
import gradio as gr
import torch
import itertools # For color cycling
import tiktoken # For GPT-4 tokenizer
from transformers import AutoTokenizer, HfArgumentParser # For Llama3 tokenizer & args potentially
import traceback # For detailed error logging
import logging # For better logging practices
from typing import Optional, Tuple, List, Dict, Any
import matplotlib.figure # For type hinting
import matplotlib.pyplot as plt

# --- Configuration ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    # Visualization
    VIZ_COLORS: List[str] = [
        "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c",
        "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a", "#ffff99", "#b15928"
    ]
    MAX_EXPECTED_SEGMENTS: int = 1 # Max segments for color map generation

    # Model/Tokenizer Names
    LLAMA3_MODEL_NAME: str = "meta-llama/Meta-Llama-3-8B" # Or choose another variant like Instruct
    TIKTOKEN_ENCODING_NAME: str = "cl100k_base"
    BLT_MODEL_NAME: str = "blt-1b" # Default Bytelatent model

    # Bytelatent Specific
    BLT_WEIGHTS_DIR: str = "hf-weights"
    BLT_MAX_BYTES_FOR_DEMO: int = 512 # Limit for this specific demo's entropy model

    # Gradio
    DEFAULT_PROMPT: str = "Daenerys Targaryen is in Game of Thrones, a fantasy epic by George R.R. Martin."
    GRADIO_THEME = gr.themes.Origin()
    GRADIO_TITLE: str = "BLT's Entropy-based Patcher vs. Tokenizer Visualisation"
    GRADIO_DESC: str = (
        "Enter text to visualize its segmentation according to different methods:\n"
        f"1.  **Byte Latent Transformer (BLT):** Entropy-based patching plot and patched text (_for this space ONLY_ - limited to ~{BLT_MAX_BYTES_FOR_DEMO} bytes using `blt_main_entropy_100m_512w`).\n"
        f"2.  **Tiktoken (GPT-4):** Text segmented by `{TIKTOKEN_ENCODING_NAME}` tokens.\n"
        f"3.  **Llama 3:** Text segmented by the `{LLAMA3_MODEL_NAME}` tokenizer."
    )

# --- Bytelatent Processor ---

# Attempt to import Bytelatent libraries
try:
    from bytelatent.data.file_util import get_fs
    from bytelatent.generate_patcher import patcher_nocache
    from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
    from bytelatent.plotting.entropy_figure_via_matplot_lib import plot_entropies
    from bytelatent.args import TrainArgs
    from download_blt_weights import main as ensure_present # Assuming this downloads weights
    _BLT_AVAILABLE = True
    logging.info("Bytelatent libraries found.")
except ImportError as e:
    logging.warning(f"Bytelatent libraries not found. Bytelatent functionality will be disabled. Error: {e}")
    _BLT_AVAILABLE = False
    # Define dummy classes/functions if BLT is not available to avoid NameErrors later
    class BltTokenizer: pass
    class TrainArgs: pass
    def patcher_nocache(*args, **kwargs): return None
    def plot_entropies(*args, **kwargs): return None
    def ensure_present(*args, **kwargs): pass
    matplotlib = None # No plotting if BLT isn't there

class BytelatentProcessor:
    """Handles loading and running the Bytelatent entropy model."""
    def __init__(self, model_name: str, weights_dir: str):
        self.model_name = model_name
        self.weights_dir = weights_dir
        self.is_available: bool = False
        self.tokenizer: Optional[BltTokenizer] = None
        self.patcher: Optional[Any] = None # Type depends on bytelatent implementation
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        if _BLT_AVAILABLE:
            try:
                # 1. Ensure weights are present
                logging.info(f"Ensuring Bytelatent model '{model_name}' weights are present...")
                ensure_present([model_name]) # Call the download script
                logging.info("Bytelatent model check complete.")

                # 2. Load Bytelatent model components
                consolidated_path = os.path.join(self.weights_dir, model_name)
                train_args_path = os.path.join(consolidated_path, "params.json")
                entropy_model_dir = os.path.join(consolidated_path, "entropy_model")

                if not os.path.exists(train_args_path):
                    raise FileNotFoundError(f"BLT training args not found at {train_args_path}.")
                if not os.path.exists(entropy_model_dir):
                     raise FileNotFoundError(f"BLT Entropy model directory not found at {entropy_model_dir}.")

                fs = get_fs(train_args_path)
                train_args = TrainArgs.model_validate_json(fs.read_text(train_args_path))

                self.tokenizer = train_args.data.tokenizer_args.build()
                assert isinstance(self.tokenizer, BltTokenizer), "Failed to build Bytelatent Tokenizer"

                patcher_args = train_args.data.patcher_args.model_copy(deep=True)
                patcher_args.realtime_patching = True
                patcher_args.patching_device = self.device
                patcher_args.device = self.device
                patcher_args.entropy_model_checkpoint_dir = entropy_model_dir
                self.patcher = patcher_args.build()

                self.is_available = True
                logging.info(f"Bytelatent processor for '{model_name}' loaded successfully on device '{self.device}'.")

            except FileNotFoundError as e:
                logging.error(f"Bytelatent setup failed: Required file/directory not found. {e}")
            except Exception as e:
                logging.error(f"An unexpected error occurred during Bytelatent setup: {e}")
                logging.error(traceback.format_exc())
        else:
             logging.warning("Skipping Bytelatent setup as libraries are unavailable.")

    def _create_highlight_data(self, patch_lengths: torch.Tensor, tokens: torch.Tensor) -> Tuple[List[Tuple[str, str]], int]:
        """Generates data for gr.HighlightedText based on bytelatent patches."""
        if not self.is_available or self.tokenizer is None or patch_lengths.numel() == 0:
            return [("Bytelatent processing failed or produced no patches.", "Error")], 0

        patch_lengths_list = patch_lengths.tolist()
        all_token_ids = tokens.tolist()
        highlighted_data = []
        current_token_index = 0
        patch_count = 0

        for i, length in enumerate(patch_lengths_list):
            if length <= 0: continue
            patch_token_ids = all_token_ids[current_token_index : current_token_index + length]
            if not patch_token_ids: continue

            try:
                patch_text = self.tokenizer.decode(patch_token_ids)
            except Exception as decode_err:
                logging.warning(f"Bytelatent patch decoding failed: {decode_err}")
                patch_text = f"[Decode Error: {len(patch_token_ids)} tokens]"

            patch_label = f"BL Patch {i+1}"
            highlighted_data.append((patch_text, patch_label))
            patch_count += 1
            current_token_index += length

        # Handle remainder tokens if any
        if current_token_index < len(all_token_ids):
            remaining_tokens = all_token_ids[current_token_index:]
            try:
                remaining_text = self.tokenizer.decode(remaining_tokens)
                label = "BL Remainder"
            except Exception:
                remaining_text = f"[Decode Error: {len(remaining_tokens)} remaining tokens]"
                label = "Error"
            highlighted_data.append((remaining_text, label))
            logging.warning(f"Bytelatent token mismatch. Consumed {current_token_index}, total {len(all_token_ids)}. Remainder added.")

        return highlighted_data, patch_count

    def process(self, prompt: str, max_bytes: int) -> Tuple[Optional[matplotlib.figure.Figure], List[Tuple[str, str]], int, str]:
        """Processes the prompt using the loaded Bytelatent model."""
        status = ""
        if not self.is_available or self.tokenizer is None or self.patcher is None:
            status = "Bytelatent processor not available."
            return None, [("Bytelatent not available.", "Error")], 0, status

        # Truncate prompt if necessary for this demo's model
        prompt_bytes = prompt.encode('utf-8')
        prompt_bl = prompt
        if len(prompt_bytes) > max_bytes:
            try:
                # Find last full character within limit (simple space split fallback)
                try:
                    prompt_bl = prompt_bytes[:max_bytes].decode('utf-8', errors='strict')
                    # If successful, find last space to avoid cutting mid-word visually
                    last_space = prompt_bl.rfind(' ')
                    if last_space != -1:
                         prompt_bl = prompt_bl[:last_space]
                except UnicodeDecodeError:
                     # If strict fails, find last valid byte sequence start before max_bytes
                     i = max_bytes
                     while i > 0:
                         try:
                             prompt_bytes[:i].decode('utf-8', errors='strict')
                             break # Found valid end point
                         except UnicodeDecodeError:
                             i -= 1
                     prompt_bl = prompt_bytes[:i].decode('utf-8', errors='ignore') # Decode ignoring errors now


                trunc_len = len(prompt_bl.encode('utf-8'))
                status = f"Warning: Prompt truncated to {trunc_len} bytes for Bytelatent entropy model.\n"
                logging.warning(status.strip())
            except Exception as trunc_err:
                 # Fallback if complex truncation fails
                 prompt_bl = prompt_bytes[:max_bytes].decode('utf-8', errors='ignore')
                 trunc_len = len(prompt_bl.encode('utf-8'))
                 status = f"Warning: Prompt aggressively truncated to ~{trunc_len} bytes due to encoding issue. Error: {trunc_err}\n"
                 logging.warning(status.strip())


        # Run Bytelatent patching
        try:
            logging.info(f"Running Bytelatent entropy model patching on {len(prompt_bl.encode('utf-8'))} bytes...")
            results = patcher_nocache([prompt_bl], tokenizer=self.tokenizer, patcher=self.patcher)
            status += "Bytelatent patching executed.\n"

            if not results:
                logging.warning("Bytelatent entropy processing returned no results.")
                status += "Warning: Bytelatent generated no patches."
                return None, [("No patches generated by Bytelatent.", "Info")], 0, status

            batch_patch_lengths, batch_scores, batch_tokens = results
            patch_lengths, scores, tokens = batch_patch_lengths[0], batch_scores[0], batch_tokens[0]

            # Create highlighted text data
            highlighted_data, patch_count = self._create_highlight_data(patch_lengths, tokens)

            # Create plot
            fig = None
            if plot_entropies is not None: # Check if plotting function is available
                try:
                    # Use the potentially truncated prompt_bl for the plot text axis if full decode fails
                    decoded_output_for_plot = self.tokenizer.decode(tokens.tolist())
                except Exception as decode_err:
                    logging.warning(f"Error decoding full BLT token sequence for plot: {decode_err}. Using (truncated) input prompt for plot axis.")
                    decoded_output_for_plot = prompt_bl

                fig = plot_entropies(patch_lengths, scores, decoded_output_for_plot, threshold=self.patcher.threshold)
                status += f"Bytelatent plot generated. Found {patch_count} patches.\n"
            else:
                 status += "Plotting unavailable.\n"

            logging.info(f"Bytelatent processing complete. Patches: {patch_count}")
            return fig, highlighted_data, patch_count, status.strip()

        except Exception as e:
            logging.error(f"An error occurred during Bytelatent processing: {e}")
            logging.error(traceback.format_exc())
            status += f"Error during Bytelatent processing: {e}"
            return None, [(f"Bytelatent Error: {e}", "Error")], 0, status.strip()


# --- Tokenizer Helpers ---

def create_tiktoken_highlight_data(prompt: str, encoding: tiktoken.Encoding) -> Tuple[List[Tuple[str, str]], int, str]:
    """Generates data for gr.HighlightedText based on tiktoken."""
    status = "Processing with Tiktoken...\n"
    try:
        tiktoken_ids = encoding.encode(prompt)
        highlighted_data = []
        for i, token_id in enumerate(tiktoken_ids):
            try:
                token_text = encoding.decode([token_id])
            except (UnicodeDecodeError, TypeError): # Handle bytes that don't form valid unicode
                try:
                    token_bytes = encoding.decode_single_token_bytes(token_id)
                    token_text = f"[Bytes: {token_bytes.hex()}]"
                except Exception: token_text = "[Decode Error]"
            except Exception as e:
                 logging.warning(f"Unexpected tiktoken decode error for token ID {token_id}: {e}")
                 token_text = "[Decode Error]"

            token_label = f"GPT4 Tk {i+1}"
            highlighted_data.append((token_text, token_label))

        token_count = len(tiktoken_ids)
        status += f"Tiktoken processing successful ({token_count} tokens)."
        logging.info(f"Tiktoken processing complete. Found {token_count} tokens.")
        return highlighted_data, token_count, status.strip()

    except Exception as e:
        logging.error(f"Error during tiktoken processing: {e}")
        logging.error(traceback.format_exc())
        status += f"Error during Tiktoken processing: {e}"
        return [(f"Error processing with tiktoken: {e}", "Error")], 0, status.strip()


def create_llama3_highlight_data(prompt: str, tokenizer: AutoTokenizer) -> Tuple[List[Tuple[str, str]], int, str]:
    """Generates data for gr.HighlightedText based on Llama 3 tokenizer."""
    status = f"Processing with Llama 3 ({tokenizer.name_or_path})...\n"
    try:
        llama_token_ids = tokenizer.encode(prompt)
        highlighted_data = []
        for i, token_id in enumerate(llama_token_ids):
            try:
                # Decode individual token. Add special handling if needed for specific tokenizers.
                token_text = tokenizer.decode([token_id])
            except Exception as e:
                logging.warning(f"Unexpected Llama 3 decode error for token ID {token_id}: {e}")
                token_text = "[Decode Error]"

            token_label = f"Llama3 Tk {i+1}"
            highlighted_data.append((token_text, token_label))

        token_count = len(llama_token_ids)
        status += f"Llama 3 processing successful ({token_count} tokens)."
        logging.info(f"Llama 3 processing complete. Found {token_count} tokens.")
        return highlighted_data, token_count, status.strip()

    except Exception as e:
        logging.error(f"Error during Llama 3 processing: {e}")
        logging.error(traceback.format_exc())
        status += f"Error during Llama 3 processing: {e}"
        return [(f"Error processing with Llama 3: {e}", "Error")], 0, status.strip()

# --- Global Initializations ---

# Initialize Bytelatent Processor (loads model if available)
blt_processor = BytelatentProcessor(Config.BLT_MODEL_NAME, Config.BLT_WEIGHTS_DIR)

# Load Tiktoken Encoding
try:
    tiktoken_encoding = tiktoken.get_encoding(Config.TIKTOKEN_ENCODING_NAME)
    logging.info(f"Tiktoken encoding '{Config.TIKTOKEN_ENCODING_NAME}' loaded.")
    tiktoken_available = True
except Exception as e:
    logging.error(f"Failed to load Tiktoken encoding '{Config.TIKTOKEN_ENCODING_NAME}': {e}")
    tiktoken_encoding = None
    tiktoken_available = False

# Load Llama 3 Tokenizer
try:
    # Use trust_remote_code=True if required by the specific model revision
    llama_tokenizer = AutoTokenizer.from_pretrained(Config.LLAMA3_MODEL_NAME) #, trust_remote_code=True)
    logging.info(f"Llama 3 tokenizer '{Config.LLAMA3_MODEL_NAME}' loaded.")
    llama_available = True
except ImportError:
    logging.error("Transformers or SentencePiece library not found. Llama 3 functionality disabled. Install with: pip install transformers sentencepiece")
    llama_tokenizer = None
    llama_available = False
except OSError as e:
    logging.error(f"Error loading Llama 3 tokenizer '{Config.LLAMA3_MODEL_NAME}': {e}")
    error_msg = f"Could not load Llama 3 tokenizer '{Config.LLAMA3_MODEL_NAME}'. Check model name, network, and authentication (use `huggingface-cli login` if needed)."
    logging.error(error_msg)
    llama_tokenizer = None
    llama_available = False
except Exception as e:
    logging.error(f"An unexpected error occurred loading Llama 3 tokenizer: {e}")
    logging.error(traceback.format_exc())
    llama_tokenizer = None
    llama_available = False


# --- Main Processing Function ---

def process_text(prompt: str) -> Tuple[
    Optional[matplotlib.figure.Figure], List[Tuple[str, str]], int, # BLT
    List[Tuple[str, str]], int, # Tiktoken
    List[Tuple[str, str]], int, # Llama 3
    str # Status
]:
    """
    Processes the input prompt using ByteLatent, Tiktoken, and Llama 3,
    returning visualizations, counts, and status.
    """
    status_messages = ["Processing started..."]
    fig = None
    bl_highlighted_data, bl_count = [("Bytelatent not available.", "Error")], 0
    tk_highlighted_data, tk_count = [("Tiktoken not available.", "Error")], 0
    llama_highlighted_data, llama_count = [("Llama 3 not available.", "Error")], 0

    # 1. Bytelatent Processing
    if blt_processor.is_available:
        fig, bl_highlighted_data, bl_count, bl_status = blt_processor.process(prompt, Config.BLT_MAX_BYTES_FOR_DEMO)
        status_messages.append(f"Bytelatent Status:\n{bl_status}")
    else:
        status_messages.append("Bytelatent Status: Skipped (processor unavailable).")

    # 2. Tiktoken Processing
    if tiktoken_available and tiktoken_encoding:
        tk_highlighted_data, tk_count, tk_status = create_tiktoken_highlight_data(prompt, tiktoken_encoding)
        status_messages.append(f"Tiktoken Status:\n{tk_status}")
    else:
        status_messages.append("Tiktoken Status: Skipped (tokenizer unavailable).")

    # 3. Llama 3 Processing
    if llama_available and llama_tokenizer:
        llama_highlighted_data, llama_count, llama_status = create_llama3_highlight_data(prompt, llama_tokenizer)
        status_messages.append(f"Llama 3 Status:\n{llama_status}")
    else:
        status_messages.append("Llama 3 Status: Skipped (tokenizer unavailable).")

    final_status = "\n---\n".join(status_messages)
    if fig is not None and matplotlib is not None:
        try:
            plt.close(fig) # Close the specific figure
            logging.debug("Closed Matplotlib figure.")
        except Exception as close_err:
            logging.warning(f"Could not close Matplotlib figure: {close_err}")
    return fig, bl_highlighted_data, bl_count, tk_highlighted_data, tk_count, llama_highlighted_data, llama_count, final_status

# --- Gradio Interface ---

def create_color_map(label_prefix: str, colors: List[str], max_segments: int) -> Dict[str, str]:
    """Generates a color map dictionary for Gradio HighlightedText."""
    color_cycler = itertools.cycle(colors)
    color_map = {f"{label_prefix} {i+1}": next(color_cycler) for i in range(max_segments)}
    color_map.update({"Error": "#FF0000", "Info": "#808080", "BL Remainder": "#AAAAAA"}) # Common labels
    return color_map

bytelatent_color_map = create_color_map("BL Patch", Config.VIZ_COLORS, Config.MAX_EXPECTED_SEGMENTS)
tiktoken_color_map = create_color_map("GPT4 Tk", Config.VIZ_COLORS, Config.MAX_EXPECTED_SEGMENTS)
llama3_color_map = create_color_map("Llama3 Tk", Config.VIZ_COLORS, Config.MAX_EXPECTED_SEGMENTS)

with gr.Blocks(theme=Config.GRADIO_THEME) as iface:
    gr.Markdown(f"# {Config.GRADIO_TITLE}")
    gr.Markdown(Config.GRADIO_DESC)

    with gr.Row():
        with gr.Column(scale=1): # Input Column
            prompt_input = gr.Textbox(
                label="Input Prompt",
                value=Config.DEFAULT_PROMPT,
                placeholder="Enter text here...",
                # Max length is for UI input; Bytelatent truncation happens in backend
                lines=5,
                info=f"Note: Bytelatent processing is limited to ~{Config.BLT_MAX_BYTES_FOR_DEMO} bytes for this demo."
            )
            submit_button = gr.Button("Generate Visualizations", variant="primary")
            status_output = gr.Textbox(label="Processing Status", interactive=False, lines=10) # More space for detailed status

        with gr.Column(scale=2): # Output Column
            # --- Bytelatent Output Area ---
            if blt_processor.is_available: # Only show BLT section if it loaded
                 with gr.Accordion("BLT Entropy Patcher Output (`blt_main_entropy_100m_512w`)", open=True):
                    with gr.Row():
                        bl_count_output = gr.Number(label="Patch Count", value=0, interactive=False, step=1, scale=0)
                    highlighted_output_bl = gr.HighlightedText(
                        label="BLT Patches",
                        color_map=bytelatent_color_map,
                        show_legend=False,
                        show_inline_category=False,
                        container=False
                    )
                    plot_output = gr.Plot(label="Entropy vs. Token Index")
            else:
                 gr.Markdown(f"### Bytelatent Output (`{Config.BLT_MODEL_NAME}`)")
                 gr.Markdown("_(Bytelatent processor failed to load or libraries are missing. Output unavailable.)_")
                 # Define dummy outputs if BLT is unavailable so the `outputs` list doesn't break
                 highlighted_output_bl = gr.HighlightedText(value=[("BLT Unavailable", "Error")], label="BLT Patches", visible=False)
                 bl_count_output = gr.Number(value=0, label="Patch Count", visible=False)
                 plot_output = gr.Plot(label="Entropy Plot", visible=False)


            # --- Tiktoken Output Area ---
            if tiktoken_available: # Only show Tiktoken section if it loaded
                with gr.Accordion(f"Tiktoken Output (`{Config.TIKTOKEN_ENCODING_NAME}`)", open=True):
                    with gr.Row():
                        tk_count_output = gr.Number(label="Token Count", value=0, interactive=False, step=1, scale=0)
                    highlighted_output_tk = gr.HighlightedText(
                        label="Tiktoken Segments",
                        color_map=tiktoken_color_map,
                        show_legend=False,
                        show_inline_category=False,
                        container=False
                    )
            else:
                gr.Markdown(f"### Tiktoken Output (`{Config.TIKTOKEN_ENCODING_NAME}`)")
                gr.Markdown("_(Tiktoken failed to load. Output unavailable.)_")
                highlighted_output_tk = gr.HighlightedText(value=[("Tiktoken Unavailable", "Error")], label="Tiktoken Segments", visible=False)
                tk_count_output = gr.Number(value=0, label="Token Count", visible=False)

            # --- Llama 3 Output Area ---
            if llama_available: # Only show Llama section if it loaded
                with gr.Accordion(f"Llama 3 Output (`{Config.LLAMA3_MODEL_NAME}`)", open=True):
                    with gr.Row():
                        llama_count_output = gr.Number(label="Token Count", value=0, interactive=False, step=1, scale=0)
                    highlighted_output_llama = gr.HighlightedText(
                        label="Llama 3 Segments",
                        color_map=llama3_color_map,
                        show_legend=False,
                        show_inline_category=False,
                        container=False
                    )
            else:
                gr.Markdown(f"### Llama 3 Output (`{Config.LLAMA3_MODEL_NAME}`)")
                gr.Markdown("_(Llama 3 tokenizer failed to load. Output unavailable.)_")
                highlighted_output_llama = gr.HighlightedText(value=[("Llama 3 Unavailable", "Error")], label="Llama 3 Segments", visible=False)
                llama_count_output = gr.Number(value=0, label="Token Count", visible=False)


    # Define the action for the button click
    submit_button.click(
        fn=process_text,
        inputs=prompt_input,
        # Ensure order matches the return values of process_text
        outputs=[
            # Bytelatent outputs (even if dummy/hidden)
            plot_output,
            highlighted_output_bl,
            bl_count_output,
            # Tiktoken outputs (even if dummy/hidden)
            highlighted_output_tk,
            tk_count_output,
            # Llama 3 outputs (even if dummy/hidden)
            highlighted_output_llama,
            llama_count_output,
            # Status output
            status_output
        ]
    )

# --- Launch the Gradio App ---
if __name__ == "__main__":
    logging.info("-----------------------------------------")
    logging.info("Starting Gradio App...")
    logging.info(f"Bytelatent Available: {blt_processor.is_available}")
    logging.info(f"Tiktoken Available: {tiktoken_available}")
    logging.info(f"Llama 3 Tokenizer Available: {llama_available}")
    logging.info("-----------------------------------------")
    iface.launch()
