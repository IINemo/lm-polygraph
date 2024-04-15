import os
import hydra
import pandas as pd
from pathlib import Path

from numpy.random import Generator, default_rng
from transformers import AutoTokenizer
from lm_polygraph.utils.dataset import Dataset

import logging

log = logging.getLogger()

hydra_config = Path(os.environ["HYDRA_CONFIG"])


def corrupt(text: str, tokenizer, gen: Generator) -> str:
    assert tokenizer is not None, "Tokenizer is not defined"
    tokens = tokenizer.encode(text)
    tokens = tokens[1:-1]  # Remove SEP and CLS tokens
    gen.shuffle(tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)


@hydra.main(
    version_base=None,
    config_path=str(hydra_config.parent),
    config_name=str(hydra_config.name),
)
def main(args):
    save_path = os.getcwd()
    log.info(f"Main directory: {save_path}")
    os.chdir(hydra.utils.get_original_cwd())

    save_path = args.save_path if "save_path" in args else save_path

    seed = args.seed if isinstance(args.seed, int) else args.seed[0]
    log.info(f"Loading dataset {args.dataset}...")
    dataset = Dataset.load(
        args.dataset,
        args.text_column,
        args.label_column,
        batch_size=args.batch_size,
        prompt=args.prompt,
        split=args.eval_split,
        load_from_disk=args.load_from_disk,
    )

    if args.subsample_eval_dataset != -1:
        dataset.subsample(args.subsample_eval_dataset, seed=seed)

    random_gen = default_rng(seed=seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    corrupted_text = [corrupt(x, tokenizer, random_gen) for x in dataset.x]

    df = pd.DataFrame(
        {
            f"{args.text_column}_corrupted": corrupted_text,
            args.text_column: dataset.x,
            args.label_column: dataset.y,
        }
    )
    df.to_csv(save_path + f"/{args.dataset}_corrupted.csv", index=False)


if __name__ == "__main__":
    main()
