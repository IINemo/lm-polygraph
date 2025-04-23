import torch
import numpy as np

from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.model_adapters.visual_whitebox_model import VisualWhiteboxModel


class GreedyLMProbsVisualCalculator(StatCalculator):
    """
    Calculates probabilities of the model generations without input texts.
    Used to calculate P(y_t|y_<t) subtrahend in PointwiseMutualInformation.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return ["greedy_lm_log_probs", "greedy_lm_log_likelihoods"], ["greedy_tokens"]

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: VisualWhiteboxModel,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the entropy of probabilities at each token position in the generation.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, consisting of:
                - 'greedy_tokens' (List[List[int]]): tokenized model generations for each input text.
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - 'greedy_lm_log_probs' (List[List[np.array]]): logarithms of autoregressive probability distributions
                    when generating the generated text without input text.
                - 'greedy_lm_log_likelihoods' (List[List[float]]): log-probabilities of generating text without input.
                    P(y_t | y_<t) for all t.
        """
        tokens = dependencies["greedy_tokens"]
        try:
            batches = {}
            for text, image in zip(texts, model.images):
                batch = model.processor_visual(
                    text=str(text),
                    images=image,
                    return_tensors="pt",
                    return_dict=True,
                )
                batch = {k: v.to(model.device()) for k, v in batch.items()}
                if not batches:
                    batches = {k: [v] for k, v in batch.items()}
                else:
                    for key in batch:
                        batches[key].append(batch[key])
            batch: Dict[str, torch.Tensor] = {
                key: torch.cat(value, dim=0) for key, value in batches.items()
            }

            with torch.no_grad():
                logprobs = model.model(**batch).logits.log_softmax(-1)
            greedy_lm_log_probs = []
            greedy_lm_ll = []
            for i in range(len(tokens)):
                if len(logprobs[i]) < len(tokens[i]):
                    raise ValueError(
                        "tokenizer(tokenizer.processor_visual.decode(t)) != t"
                    )
                greedy_lm_log_probs.append(
                    logprobs[i, -len(tokens[i]) : -1].cpu().numpy()
                )
                greedy_lm_ll.append(
                    [
                        logprobs[i, -len(tokens[i]) + j, tokens[i][j]].item()
                        for j in range(len(tokens[i]))
                    ]
                )
        except ValueError:
            # case where tokenizer(tokenizer.decode(t)) != t; process each sequence separetly
            greedy_lm_log_probs = []
            greedy_lm_ll = []
            for toks in tokens:
                input_ids = torch.LongTensor([toks]).to(model.device())
                batch = {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids).to(model.device()),
                }
                with torch.no_grad():
                    if model.model_type == "Seq2SeqLM":
                        logprobs = model.model(
                            **batch, decoder_input_ids=batch["input_ids"]
                        ).logits.log_softmax(-1)
                    else:
                        logprobs = model.model(**batch).logits.log_softmax(-1)
                logprobs = logprobs[0]
                assert len(logprobs) >= len(toks)
                greedy_lm_log_probs.append(logprobs[-len(toks) : -1].cpu().numpy())
                greedy_lm_ll.append(
                    [logprobs[-len(toks) + j, toks[j]].item() for j in range(len(toks))]
                )
        return {
            "greedy_lm_log_probs": greedy_lm_log_probs,
            "greedy_lm_log_likelihoods": greedy_lm_ll,
        }
