import torch
import numpy as np

from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel, BlackboxModel
from lm_polygraph.stat_calculators.embeddings import get_embeddings_from_output


class BlackboxSamplingGenerationCalculator(StatCalculator):
    """
    Calculates several sampled texts for Blackbox model (lm_polygraph.BlackboxModel).
    """

    def __init__(self, samples_n: int = 10):
        """
        Parameters:
            samples_n (int): number of samples to generate per input text. Default: 10
        """
        self.samples_n = samples_n
        super().__init__(["sample_texts"], [])

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: BlackboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates sampled texts for Blackbox model on the input batch.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, can be empty (not used).
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with List[List[str]] sampled texts at 'sample_texts' key.
        """

        if isinstance(model, BlackboxModel):
            samples = model.generate_texts(
                input_texts=texts,
                max_new_tokens=max_new_tokens,
                n=self.samples_n,
            )
        else:
            samples = [[] for _ in range(len(texts))]
            out = model.generate_texts(
                input_texts=texts,
                max_new_tokens=max_new_tokens,
                min_length=2,
                do_sample=True,
                num_beams=1,
                num_return_sequences=self.samples_n,
            )
            for i in range(len(texts)):
                for j in range(self.samples_n):
                    samples[i].append(out[i * self.samples_n + j])

        return {
            "sample_texts": samples,
        }


def _gen_samples(n_samples, model, batch, **kwargs):
    batch_size = len(batch["input_ids"])
    logits, sequences = [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]
    mean_embeddings = [[] for _ in range(batch_size)]
    last_embeddings = [[] for _ in range(batch_size)]

    with torch.no_grad():
        for k in range(n_samples):
            out = model.generate(**batch, **kwargs)
            cur_logits = torch.stack(out.scores, dim=1)
            
            _, _, mean_emb, last_emb = get_embeddings_from_output(
                out,
                batch,
                model.model_type,
                model.tokenizer.pad_token_id
            )

            for i in range(batch_size):
                sequences[i].append(out.sequences[i])
                logits[i].append(cur_logits[i])
                mean_embeddings[i].append(mean_emb[i])
                last_embeddings[i].append(last_emb[i])

    sequences = [s for sample_seqs in sequences for s in sample_seqs]
    return sequences, sum(logits, []), mean_embeddings, last_embeddings


class SamplingGenerationCalculator(StatCalculator):
    """
    For Whitebox model (lm_polygraph.WhiteboxModel), at input texts batch calculates:
    * sampled texts
    * tokens of the sampled texts
    * probabilities of the sampled tokens generation
    """

    def __init__(self, samples_n: int = 10, n_alternatives: int = 10):
        """
        Parameters:
            samples_n (int): number of samples to generate per input text. Default: 10
        """
        self.samples_n = samples_n
        self.n_alternatives = n_alternatives
        super().__init__(
            [
                "sample_log_probs",
                "sample_tokens",
                "sample_texts",
                "sample_log_likelihoods",
                "sample_tokens_distributions",
                "sample_tokens_alternatives",
                "sample_mean_all_layers_embeddings",
                "sample_last_all_layers_embeddings",
            ],
            [],
        )

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the statistics of sampling texts.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, can be empty (not used).
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - 'sample_texts' (List[List[str]]): `samples_n` texts for each input text in the batch,
                - 'sample_tokens' (List[List[List[float]]]): tokenized 'sample_texts',
                - 'sample_log_probs' (List[List[float]]): sum of the log probabilities at each token of the sampling generation.
                - 'sample_log_likelihoods' (List[List[List[float]]]): log probabilities at each token of the sampling generation.
                - 'token_distributions' (List[List[List[float]]]): full token probability distributions for each generated token.
        """
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        sequences, logits, mean_embeddings, last_embeddings = _gen_samples(
            self.samples_n,
            model,
            batch,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=max_new_tokens,
            min_new_tokens=2,
            output_attentions=False,
            output_hidden_states=True,
            do_sample=True,
            num_beams=1,
            num_return_sequences=1,
            suppress_tokens=(
                []
                if model.generation_parameters.allow_newlines
                else [
                    t
                    for t in range(len(model.tokenizer))
                    if "\n" in model.tokenizer.decode([t])
                ]
            ),
        )

        log_probs = [[] for _ in range(len(texts))]
        tokens = [[] for _ in range(len(texts))]
        texts = [[] for _ in range(len(texts))]
        log_likelihoods = [[] for _ in range(len(texts))]
        token_distributions = [[] for _ in range(len(texts))]
        alternatives = [[] for _ in range(len(texts))]


        if model.model_type == "Seq2SeqLM":
            sequences = [seq[1:] for seq in sequences]

        for i in range(len(logits)):
            log_prob, ll, toks, distributions = 0, [], [], []
            inp_size = (
                len(batch["input_ids"][int(i / self.samples_n)])
                if model.model_type == "CausalLM"
                else 0
            )
            gen_size = len(sequences[i]) - inp_size
            sample_alternatives = [[] for _ in range(gen_size)]
            for j in range(gen_size):
                cur_token = sequences[i][j + inp_size].item()
                log_prob += logits[i][j][cur_token].item()
                ll.append(logits[i][j][cur_token].item())
                toks.append(cur_token)

                lt = logits[i][j].cpu().numpy()
                distributions.append(lt)

                best_tokens = np.argpartition(lt, -self.n_alternatives)
                ln = len(best_tokens)
                best_tokens = best_tokens[ln - self.n_alternatives : ln]
                for t in best_tokens:
                    sample_alternatives[j].append((t.item(), lt[t].item()))
                sample_alternatives[j].sort(
                    key=lambda x: x[0] == cur_token,
                    reverse=True,
                )

            log_likelihoods[int(i / self.samples_n)].append(ll)
            log_probs[int(i / self.samples_n)].append(log_prob)
            tokens[int(i / self.samples_n)].append(toks)
            texts[int(i / self.samples_n)].append(model.tokenizer.decode(toks, skip_special_tokens=True))
            token_distributions[int(i / self.samples_n)].append(distributions)
            alternatives[int(i / self.samples_n)].append(sample_alternatives)

        return {
            "sample_log_likelihoods": log_likelihoods,
            "sample_log_probs": log_probs,
            "sample_tokens": tokens,
            "sample_texts": texts,
            "sample_tokens_distributions": token_distributions,
            "sample_tokens_alternatives": alternatives,
            "sample_mean_all_layers_embeddings": mean_embeddings,
            "sample_last_all_layers_embeddings": last_embeddings,
        }

class FirstSampleCalculator(StatCalculator):
    def __init__(self):
        super().__init__(
            [
                "first_sample_texts",
            ],
            [
                "sample_texts",
            ]
        )

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        sample_texts = dependencies["sample_texts"]
        first_sample_texts = [st[0] for st in sample_texts]

        return {
            "first_sample_texts": first_sample_texts,
        }

class BestSampleCalculator(StatCalculator):
    def __init__(self):
        super().__init__(
            [
                "best_sample_texts",
                "best_sample_text_ids",
                "best_normalized_sample_texts",
                "best_normalized_sample_text_ids",
                "best_mean_all_layers_embeddings",
                "best_last_all_layers_embeddings",
            ],
            [
                "sample_texts",
                "sample_log_probs",
                "sample_log_likelihoods",
                "sample_mean_all_layers_embeddings",
                "sample_last_all_layers_embeddings",
            ]
        )

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        best_sample_texts = []
        best_sample_text_ids = []
        best_normalized_sample_texts = []
        best_normalized_sample_text_ids = []
        best_mean_embeddings = []
        best_last_embeddings = []

        for batch_i, (sample_texts, sample_log_probs, sample_log_likelihoods, mean_embs, last_embs) in enumerate(zip(
            dependencies["sample_texts"], 
            dependencies["sample_log_probs"], 
            dependencies["sample_log_likelihoods"],
            dependencies["sample_mean_all_layers_embeddings"],
            dependencies["sample_last_all_layers_embeddings"],
        )):
            best_i = np.argmax(sample_log_probs)
            best_sample_texts.append(sample_texts[best_i])
            best_sample_text_ids.append(best_i)
            best_mean_embeddings.append(mean_embs[best_i])
            best_last_embeddings.append(last_embs[best_i])

            ppls = [np.mean(ll) for ll in sample_log_likelihoods]
            best_ppl_i = np.argmax(ppls)
            best_normalized_sample_texts.append(sample_texts[best_ppl_i])
            best_normalized_sample_text_ids.append(best_ppl_i)

        return {
            "best_sample_texts": best_sample_texts,
            "best_sample_text_ids": best_sample_text_ids,
            "best_normalized_sample_texts": best_normalized_sample_texts,
            "best_normalized_sample_text_ids": best_normalized_sample_text_ids,
            "best_mean_all_layers_embeddings": best_mean_embeddings,
            "best_last_all_layers_embeddings": best_last_embeddings,
        }
