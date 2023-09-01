from lm_polygraph.utils.dataset import Dataset

from numpy.random import Generator


def corrupt(text: str, tokenizer, gen: Generator) -> str:
    assert tokenizer is not None, "Tokenizer is not defined"
    tokens = tokenizer.encode(text)
    tokens = tokens[1:-1]  # Remove SEP and CLS tokens
    gen.shuffle(tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)


def add_ood_from_dataset(
    dataset: Dataset,
    ood: Dataset,
    random_gen: Generator,
    corrupt_samples: bool = False,
    text_field: str = "text",
    ood_text_field: str = "text",
    remove_other_columns: bool = True,
    tokenizer=None,
) -> Dataset:
    if len(ood) > len(dataset):
        ood = ood.subsample(size=len(dataset.x), seed=42)

    if corrupt_samples:
        ood.x = [corrupt(x, tokenizer, random_gen) for x in ood.x]
        
    ood_label = [0]*len(dataset.x) + [1]*len(ood.x)
    
    test_dataset = Dataset(x=dataset.x+ood.x,
                            y=dataset.y+ood.y,
                            batch_size=dataset.batch_size)
    test_dataset.ood_label = ood_label
    return test_dataset
