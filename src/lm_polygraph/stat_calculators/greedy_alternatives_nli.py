import numpy as np

from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.deberta import Deberta
from collections import defaultdict
import torch.nn as nn
import string
import torch

from cachetools import LRUCache


def _eval_nli_model(nli_queue: List[Tuple[str, str]], deberta: Deberta) -> List[str]:
    nli_set = list(set(nli_queue))

    softmax = nn.Softmax(dim=1)
    w_probs = defaultdict(lambda: defaultdict(lambda: None))
    for k in range(0, len(nli_set), deberta.batch_size):
        batch = nli_set[k: k + deberta.batch_size]
        encoded = deberta.deberta_tokenizer.batch_encode_plus(
            batch, padding=True, return_tensors="pt"
        ).to(deberta.device)
        with torch.inference_mode():
            with torch.amp.autocast(device_type='cuda'):
                logits = deberta.deberta(**encoded).logits.float().detach()
        for (wi, wj), prob in zip(batch, softmax(logits).cpu().detach()):
            w_probs[wi][wj] = prob

    classes = []
    for w1, w2 in nli_queue:
        pr = w_probs[w1][w2]
        id = pr.argmax()
        ent_id = deberta.deberta.config.label2id["ENTAILMENT"]
        contra_id = deberta.deberta.config.label2id["CONTRADICTION"]
        if id == ent_id:
            str_class = "entail"
        elif id == contra_id:
            str_class = "contra"
        else:
            str_class = "neutral"
        classes.append(str_class)
    return classes


class NLIEvaluator:
    def __init__(self, deberta, max_cache_size: int = 100_000):
        self.deberta = deberta
        self.max_cache_size = max_cache_size
        # Get entailment / contradiction label ids
        self.ent_id = self.deberta.deberta.config.label2id["ENTAILMENT"]
        self.contra_id = self.deberta.deberta.config.label2id["CONTRADICTION"]
    
    def _run_inference(self, pairs: List[Tuple[str, str]]) -> List[Tuple[Tuple[str, str], str]]:
        """
        Perform NLI inference on the given text pairs
        
        Args:
            pairs: List of text pairs to perform inference on
            
        Returns:
            List of inference results, each element is (text_pair, label)
        """
        if not pairs:
            return []
            
        softmax = nn.Softmax(dim=1)
        results = []
        
        # Batch processing
        for i in range(0, len(pairs), self.deberta.batch_size):
            batch = pairs[i: i + self.deberta.batch_size]
            
            encoded = self.deberta.deberta_tokenizer.batch_encode_plus(
                batch, padding=True, return_tensors="pt"
            ).to(self.deberta.device)
            
            with torch.inference_mode():
                with torch.amp.autocast(device_type='cuda'):
                    logits = self.deberta.deberta(**encoded).logits.float().detach()
            
            # Calculate probabilities and get labels
            probs = softmax(logits).cpu()
            
            for pair, prob in zip(batch, probs):
                pred_id = prob.argmax().item()
                if pred_id == self.ent_id:
                    label = "entail"
                elif pred_id == self.contra_id:
                    label = "contra"
                else:
                    label = "neutral"
                results.append((pair, label))
                
        return results

    def __call__(self, nli_queue: List[Tuple[str, str]]) -> List[str]:
        self.cache = LRUCache(maxsize=self.max_cache_size)
        # 1. Set default cache for self-entailing pairs (x, x)
        for pair in nli_queue:
            if pair[0] == pair[1] and pair not in self.cache:
                self.cache[pair] = "entail"

        # 2. Find uncached pairs (preserve order + no deduplication)
        uncached_pairs = [pair for pair in nli_queue if pair not in self.cache]

        if uncached_pairs:
            # 3. Use the abstracted inference function to process uncached pairs
            inference_results = self._run_inference(uncached_pairs)
            
            # 4. Write results to cache
            for pair, label in inference_results:
                self.cache[pair] = label

        # 5. Read cache and return results, run inference again for any still uncached pairs
        results = []
        still_uncached = []
        
        for pair in nli_queue:
            if pair not in self.cache:
                still_uncached.append(pair)
                
        if still_uncached:
            print(f"Warning: {len(still_uncached)} pairs not in cache after processing, running model again")
            inference_results = self._run_inference(still_uncached)
            for pair, label in inference_results:
                self.cache[pair] = label
                
        # Now all pairs should be in the cache
        results = [self.cache[pair] for pair in nli_queue]
        return results


class GreedyAlternativesNLICalculator(StatCalculator):
    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return ["greedy_tokens_alternatives_nli"], ["greedy_tokens_alternatives"]

    def __init__(self, nli_model, batch_size: int = 10):
        super().__init__()
        self.nli_model = nli_model
        self.batch_size = batch_size
        self.nli_model.batch_size = self.batch_size

    def _strip(self, w: str):
        return w.strip(string.punctuation + " \n")

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        greedy_alternatives = dependencies["greedy_tokens_alternatives"]
        self.nli_model.batch_size = self.batch_size

        # Step 1: Collect all nli_queue items from all samples
        all_nli_queue = []
        queue_meta = []  # (sample_idx, word_idx, word_alternatives, words)

        for sample_idx, sample_alternatives in enumerate(greedy_alternatives):
            for word_idx, word_alternatives in enumerate(sample_alternatives):
                if len(word_alternatives) > 0 and not isinstance(word_alternatives[0][0], str):
                    word_alternatives = [
                        (model.tokenizer.decode([alt]), prob)
                        for alt, prob in word_alternatives
                    ]
                words = [self._strip(alt[0]) for alt in word_alternatives]
                queue_meta.append(
                    (sample_idx, word_idx, word_alternatives, words))
                for wi in words:
                    all_nli_queue.append((words[0], wi))
                    all_nli_queue.append((wi, words[0]))

        # Step 2: Call NLI evaluator once
        if not hasattr(self, "_nli_evaluator"):
            self._nli_evaluator = NLIEvaluator(self.nli_model)
        all_nli_classes = self._nli_evaluator(all_nli_queue)

        # Step 3: Fill results
        greedy_alternatives_nli = [[] for _ in greedy_alternatives]
        ptr = 0

        for sample_idx, word_idx, word_alternatives, words in queue_meta:
            L = len(words)
            nli_matrix = [["" for _ in range(L)] for _ in range(L)]
            nli_class = defaultdict(lambda: None)

            for _ in range(2 * L):
                w1, w2 = all_nli_queue[ptr]
                label = all_nli_classes[ptr]
                nli_class[w1, w2] = label
                ptr += 1

            for i, wi in enumerate(words):
                for j, wj in enumerate(words):
                    if i > 0 and j > 0:
                        continue
                    nli_matrix[i][j] = nli_class[wi, wj]

            greedy_alternatives_nli[sample_idx].append(nli_matrix)

        return {"greedy_tokens_alternatives_nli": greedy_alternatives_nli}


class GreedyAlternativesFactPrefNLICalculator(StatCalculator):
    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the SamplingPromptCalculator.
        """

        return ["greedy_tokens_alternatives_fact_pref_nli"], [
            "greedy_tokens_alternatives",
            "greedy_tokens",
            "claims",
        ]

    def __init__(self, nli_model):
        super().__init__()
        self.nli_model = nli_model

    def _strip(self, w: str):
        return w.strip(string.punctuation + " \n")

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        greedy_alternatives = dependencies["greedy_tokens_alternatives"]
        greedy_tokens = dependencies["greedy_tokens"]
        claims = dependencies["claims"]
        greedy_alternatives_nli = []
        for sample_alternatives, sample_claims, sample_tokens in zip(
            greedy_alternatives,
            claims,
            greedy_tokens,
        ):
            nli_queue = []
            for claim in sample_claims:
                tokens = [sample_tokens[t] for t in claim.aligned_token_ids]
                alts = [sample_alternatives[t]
                        for t in claim.aligned_token_ids]
                for i in range(len(tokens)):
                    for j in range(len(alts[i])):
                        text1 = model.tokenizer.decode(tokens[: i + 1])
                        text2 = model.tokenizer.decode(
                            tokens[:i] + [alts[i][j][0]])
                        nli_queue.append((text1, text2))
                        nli_queue.append((text2, text1))

            nli_classes = _eval_nli_model(nli_queue, self.nli_model)

            nli_matrixes = []
            for claim in sample_claims:
                nli_matrixes.append([])
                tokens = [sample_tokens[t] for t in claim.aligned_token_ids]
                alts = [sample_alternatives[t]
                        for t in claim.aligned_token_ids]
                for i in range(len(tokens)):
                    nli_matrix = []
                    for _ in range(len(alts[i])):
                        nli_matrix.append([])
                        for j in range(len(alts[i])):
                            nli_matrix[-1].append(None)
                    for j in range(len(alts[i])):
                        nli_matrix[0][j], nli_matrix[j][0] = nli_classes[:2]
                        nli_classes = nli_classes[2:]
                    nli_matrixes[-1].append(nli_matrix)
            greedy_alternatives_nli.append(nli_matrixes)

        return {"greedy_tokens_alternatives_fact_pref_nli": greedy_alternatives_nli}
