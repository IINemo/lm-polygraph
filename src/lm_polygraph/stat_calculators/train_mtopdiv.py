import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from typing import Dict, List, Tuple, Literal, Callable
from itertools import product

from ..stat_calculators import (
    StatCalculator,
    GreedyProbsCalculator,
    AttentionForwardPassCalculator,
)
from ..generation_metrics import RougeMetric
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.estimators.mtopdiv_utils import (
    get_mtopdivs,
    load_model_heads,
    save_model_heads,
)


class TrainMTopDivCalculator(StatCalculator):
    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return ["topological_divergence_heads"], []

    def __init__(
        self,
        priority: Literal["train", "cache"] = "cache",
        load_train_dataset_fn: Callable = None,
        cache_path: str = None,
        max_heads: int = 6,
        n_jobs: int = -1,
    ):
        super().__init__()

        self.priority = priority
        self.cache_path = cache_path
        self.load_train_dataset_fn = load_train_dataset_fn
        self.max_heads = max_heads
        self.n_jobs = n_jobs

    def select_heads(self, scores, labels):
        grounded_scores, hal_scores = scores[labels == 0], scores[labels == 1]
        deltas = hal_scores.mean(0) - grounded_scores.mean(0)
        heads = sorted(range(len(deltas)), key=lambda x: deltas[x], reverse=True)

        best_auroc, n_opt = 0, 0
        for n in range(1, self.max_heads + 1):
            n_best_heads = heads[:n]
            predictions = scores[:, n_best_heads].mean(axis=1)
            roc_auc = roc_auc_score(labels, predictions)
            if roc_auc > best_auroc:
                best_auroc = roc_auc
                n_opt = n
        return heads[:n_opt]

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        name_or_path = model.model.config.name_or_path
        heads = load_model_heads(self.cache_path, name_or_path)
        if heads and self.priority == "cache":
            heads = np.array(heads)
            heads = heads[: self.max_heads]
            return {"topological_divergence_heads": heads}

        train_dataset = self.load_train_dataset_fn()

        mtopdivs = []
        labels = []

        greedy_calc = GreedyProbsCalculator(False, False)
        attn_forward_pass_calc = AttentionForwardPassCalculator()
        generation_metric = RougeMetric("rougeL")

        for input_text, target in tqdm(train_dataset):
            stats = greedy_calc(
                dependencies=None,
                texts=input_text,
                model=model,
                max_new_tokens=max_new_tokens,
            )
            attn_weights_batch = attn_forward_pass_calc(
                dependencies=stats,
                texts=input_text,
                model=model,
                max_new_tokens=max_new_tokens,
            )["forwardpass_attention_weights"]
            length_responses = list(map(len, stats["greedy_tokens"]))

            _, num_layers, num_heads, _, _ = attn_weights_batch.shape
            heads = product(range(num_layers), range(num_heads))

            mtopdivs.append(
                get_mtopdivs(
                    heads,
                    length_responses,
                    attn_weights_batch,
                    n_jobs=self.n_jobs,
                )
            )
            labels.append(
                generation_metric(
                    stats=stats,
                    target_texts=target,
                )
            )

        mtopdivs = np.concatenate(mtopdivs, axis=0)

        labels = np.concatenate(labels)
        labels = ~(labels > 0.3)

        heads = self.select_heads(mtopdivs, labels)
        heads = np.unravel_index(heads, (num_layers, num_heads))
        heads = np.stack(heads, axis=1)

        save_model_heads(self.cache_path, name_or_path, heads.tolist())
        return {"topological_divergence_heads": heads}
