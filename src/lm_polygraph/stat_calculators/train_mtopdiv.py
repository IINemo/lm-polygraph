import gc
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from typing import Dict, List, Tuple, Literal
from itertools import product

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.estimators.mtopdiv import (
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

    def __init__(self, 
                 priority: Literal["train", "cache"] = "cache", 
                 train_dataset: "MTopDivHoldoutDataset" = None, 
                 cache_path: str = None, 
                 max_heads: int = 6,
                 n_jobs: int = -1):
        super().__init__()
        if train_dataset is None and cache_path is None:
            raise Exception("Either train_dataset or cache_path must be provided.")
        self.priority = priority
        self.cache_path = cache_path
        self.train_dataset = train_dataset
        self.max_heads = max_heads
        self.n_jobs = n_jobs

    @staticmethod
    def calculate_attention_forward_pass(
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
            mask = mask[:, None, None, :] & mask[:, None, :, None]
            attns = [attn.masked_fill(mask == 0, float("nan")).cpu() for attn in attns]
        
        torch.cuda.empty_cache()
        gc.collect()

        return torch.stack(attns, dim=1).numpy()
    
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
    
    def train(self, model, train_dataloader):
        train_mtopdivs = []
        train_labels = []

        for inp_texts, target_texts, labels in tqdm(train_dataloader):
            length_responses = model.tokenizer(
                target_texts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            length_responses = length_responses["attention_mask"].sum(dim=1).tolist()
            forwardpass_attention_weights = self.calculate_attention_forward_pass(
                inp_texts,
                target_texts,
                model,
            )
            _, num_layers, num_heads, _, _ = forwardpass_attention_weights.shape
            heads = product(range(num_layers), range(num_heads))
            mtopdivs = get_mtopdivs(
                heads,
                length_responses,
                forwardpass_attention_weights,
                self.n_jobs
            )
            
            train_mtopdivs.append(mtopdivs)
            train_labels += labels

            del forwardpass_attention_weights
            gc.collect()

        train_mtopdivs = np.concatenate(train_mtopdivs, axis=0)
        train_labels = np.array(train_labels)
        return {
            "train_mtopdivs": train_mtopdivs,
            "train_labels": train_labels,
            "num_layers": num_layers,
            "num_heads": num_heads,
        }

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        heads = load_model_heads(self.cache_path, model.model_path)
        if heads and self.priority == "cache":
            heads = np.array(heads)
            return {"topological_divergence_heads": heads[:self.max_heads]}
        
        self.train_dataset.from_csv()
        self.train_dataset.subsample()
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.train_dataset.batch_size,
            shuffle=True
        )
        output = self.train(model, train_dataloader)
        heads = self.select_heads(
            output["train_mtopdivs"],
            output["train_labels"],
        )
        heads = np.unravel_index(heads, (output["num_layers"], output["num_heads"]))
        heads = np.stack(heads, axis=1)

        save_model_heads(self.cache_path, model.model_path, heads.tolist())
        return {"topological_divergence_heads": heads}
