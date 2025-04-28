import os

import typer

from bytelatent.data.file_util import get_fs
from bytelatent.distributed import DistributedArgs, setup_torch_distributed
from bytelatent.generate_patcher import patcher_nocache
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
from bytelatent.plotting.entropy_figure_via_matplot_lib import plot_entropies


def main(prompt: str, model_name: str = "blt-1b"):
    from bytelatent.args import TrainArgs
    consolidated_path = os.path.join("hf-weights", model_name)
    train_args_path = os.path.join(consolidated_path, "params.json")
    fs = get_fs(train_args_path)
    train_args = TrainArgs.model_validate_json(fs.read_text(train_args_path))

    tokenizer = train_args.data.tokenizer_args.build()
    assert isinstance(tokenizer, BltTokenizer)
    patcher_args = train_args.data.patcher_args.model_copy(deep=True)
    patcher_args.realtime_patching = True
    #  NOTE: CPU currently unsupported due to reliance of xformers
    patcher_args.patching_device = "cpu"
    patcher_args.device = "cpu"
    print("Loading entropy model and patcher")
    patcher_args.entropy_model_checkpoint_dir = os.path.join(
        consolidated_path, "entropy_model"
    )
    patcher = patcher_args.build()
    prompts = [prompt]
    results = patcher_nocache(
        prompts, tokenizer=tokenizer, patcher=patcher
    )
    if not results:
        raise Exception("Ruh roh")
    batch_patch_lengths, batch_scores, batch_tokens = results
    decoded_chars = [tokenizer.decode(row_tokens.tolist()) for row_tokens in batch_tokens]
    plot_entropies(
        batch_patch_lengths[0],
        batch_scores[0],
        decoded_chars[0],
        threshold=patcher.threshold
    )


if __name__ == "__main__":
    typer.run(main)
