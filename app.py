import os
import gradio as gr
import torch
# Removed matplotlib and plot_entropies imports

# Assuming bytelatent library and its dependencies are installed
from bytelatent.data.file_util import get_fs
# from bytelatent.distributed import DistributedArgs, setup_torch_distributed # Not needed
from bytelatent.generate_patcher import patcher_nocache
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
# Removed: from bytelatent.plotting.entropy_figure_via_matplot_lib import plot_entropies
from bytelatent.args import TrainArgs
from download_blt_weights import main as ensure_present

# --- Global Setup (Consider loading models outside if necessary) ---
# Kept inside the function for simplicity as before.

def process_text(prompt: str, model_name: str = "blt-1b"):
    """
    Processes the input prompt using the ByteLatent model and returns decoded characters.

    Args:
        prompt: The input text string from the Gradio interface.
        model_name: The name of the model to use.

    Returns:
        A string containing the decoded characters after processing, or an error message.
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
            return "Processing completed, but no results were generated." # Return info message

        batch_patch_lengths, batch_scores, batch_tokens = results
        # Decode the first (and only) result in the batch
        decoded_chars_list = [tokenizer.decode(row_tokens.tolist()) for row_tokens in batch_tokens]
        decoded_output = decoded_chars_list[0] if decoded_chars_list else "No characters decoded."

        print("Processing and decoding complete.")
        # --- End Processing ---


        # Return the decoded text string
        return decoded_output

    except FileNotFoundError as e:
        print(f"Error: {e}")
        # raise gr.Error(str(e)) # Display specific error in Gradio UI
        return f"Error: {str(e)}" # Return error as text output
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        # raise gr.Error(f"An error occurred during processing: {e}")
        return f"An unexpected error occurred: {e}" # Return error as text output


# --- Gradio Interface Definition ---
iface = gr.Interface(
    fn=process_text,
    inputs=gr.Textbox(
        label="Input Prompt",
        placeholder="Enter your text here..."
    ),
    # Changed output to display the decoded text
    outputs=gr.Text(label="Decoded Output"),
    title="ByteLatent Text Processor",
    description="Enter text to process it with the ByteLatent model ('blt-1b' by default). The decoded output will be shown.",
    allow_flagging="never",
)

# --- Launch the Gradio App ---
if __name__ == "__main__":
    ensure_present(["blt-1b"])
    iface.launch()
