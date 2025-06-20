import gc
import torch
import numpy as np
from tqdm import tqdm

from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
from .attention_forward_pass import AttentionForwardPassCalculator
from ..estimators.mtopdiv import (
    transform_attention_scores_to_distances,
    transform_distances_to_mtopdiv,
)


class TrainMTopDivCalculator(StatCalculator):
    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return [
            "train_labels",
            "train_mtopdivs",
        ], []

    def __init__(self, train_dataset=None, *args, **kwargs):
        super().__init__()
        self.train_dataset = train_dataset
        self.base_calculator = AttentionForwardPassCalculator()

    def _calculate_attention_forward_pass(
        self,
        prompts: List[str],
        responses: List[str],
        model: WhiteboxModel,
    ) -> Tuple[List[np.ndarray], np.ndarray]:

        if model.tokenizer.chat_template is not None:
            chats = [
                [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
                for p, r in zip(prompts, responses)
            ]
        else:
            chats = [p + r for p, r in zip(prompts, responses)]

        with torch.no_grad():
            batch = model.tokenize(chats).to(model.device())
            attns = model.model(
                **batch,
                output_attentions=True,
            ).attentions
            mask = batch["attention_mask"]
            mask = mask[:, None, None, None, :] & mask[:, None, None, :, None]
            attns = torch.stack(attns, dim=1)
            attns = attns.masked_fill(mask == 0, float("nan")).cpu()

        torch.cuda.empty_cache()
        gc.collect()

        return attns

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:

        train_mtopdivs = []
        train_labels = []

        for inp_texts, target_texts, labels in tqdm(self.train_dataset):
            forwardpass_attention_weights = self._calculate_attention_forward_pass(
                inp_texts,
                target_texts,
                model,
            )

            length_responses = model.tokenizer(
                target_texts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            length_responses = length_responses["attention_mask"].sum(dim=1).tolist()

            batch_size, layers, heads, seq_len, _ = forwardpass_attention_weights.shape
            paddings = np.isnan(forwardpass_attention_weights[:, 0, 0, 0]).sum(axis=-1)
            distance_matrices_batch = transform_attention_scores_to_distances(
                forwardpass_attention_weights
            )
            distance_matrices_batch = distance_matrices_batch.reshape(
                batch_size, layers * heads, seq_len, seq_len
            ).numpy()

            def job(head):
                mtopdivs = []
                for sample_id in range(batch_size):
                    distance_matrice = distance_matrices_batch[sample_id, head]
                    padding = paddings[sample_id]
                    response_length = length_responses[sample_id]
                    if padding > 0:
                        distance_matrice = distance_matrice[:-padding, :-padding]
                    distance_matrice[:-response_length, :-response_length] = 0
                    mtopdiv = transform_distances_to_mtopdiv(distance_matrice)
                    mtopdivs.append(mtopdiv)
                return np.array(mtopdivs, dtype=float)

            train_mtopdivs.append(
                np.stack(
                    [job(head) for head in range(heads * layers)],
                    axis=1,
                )
            )
            train_labels += labels

        train_mtopdivs = np.concatenate(train_mtopdivs, axis=0)
        train_labels = np.array(train_labels)

        return {
            "train_mtopdivs": train_mtopdivs,
            "train_labels": train_labels,
        }
