import numpy as np
import os
import torch

from typing import Dict, List
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from .estimator import Estimator


def split_classes(
    tokens: List[str],
    batch_size: int,
    semantic_bert_path: str = "sentence-transformers/bert-base-nli-mean-tokens",
    sim_threshold: float = 0.85,
) -> np.ndarray:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(semantic_bert_path)
    model = AutoModel.from_pretrained(semantic_bert_path).to(device)
    classes_embeddings: np.ndarray = np.zeros(
        shape=(0, model.pooler.dense.out_features)
    )

    classes_sizes: List[int] = []
    sample_to_class: List[int] = []

    rng = tqdm(
        range(0, len(tokens), batch_size),
        total=(len(tokens) + batch_size - 1) // batch_size,
    )
    for i in rng:
        batch = tokenizer(tokens[i : i + batch_size], return_tensors="pt", padding=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            state = model(**batch).last_hidden_state
            embeddings = (
                torch.vstack(
                    [
                        l[attn == 1].mean(0)
                        for attn, l in zip(batch["attention_mask"], state)
                    ]
                )
                .cpu()
                .numpy()
            )
        for j in range(len(embeddings)):
            if len(classes_embeddings) != 0:
                sims = cosine_similarity([embeddings[j]], classes_embeddings)
            else:
                sims = []
            if len(sims) == 0 or sims.max() < sim_threshold:
                classes_embeddings = np.append(
                    classes_embeddings, embeddings[j].reshape(1, -1), axis=0
                )
                sample_to_class.append(len(classes_sizes))
                classes_sizes.append(0)
            else:
                cl = sims.argmax()
                classes_embeddings[cl] = (
                    classes_embeddings[cl] * classes_sizes[cl] + embeddings[j]
                ) / (classes_sizes[cl] + 1)
                classes_sizes[cl] += 1
                sample_to_class.append(cl)

        rng.set_description(
            f"{min(i + batch_size, len(tokens))} tokens, {len(classes_sizes)} classes"
        )

    return np.array(sample_to_class)


class SemanticEntropyToken(Estimator):
    def __init__(
        self,
        tokenizer_path: str,
        tokenizer_save_path: str,
        semantic_bert_path: str = "sentence-transformers/bert-base-nli-mean-tokens",
        batch_size: int = 10,
    ):
        super().__init__(["greedy_log_probs"], "token")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, padding_side="left", add_bos_token=True
        )
        tokens = [tokenizer.decode([i]) for i in range(len(tokenizer))]
        tokenizer_classes_path = os.path.join(
            tokenizer_save_path, tokenizer.name_or_path.split("/")[-1] + "_classes.npy"
        )
        if os.path.exists(tokenizer_classes_path):
            print(f"Loading tokenizer classes from {tokenizer_classes_path}")
            self.classes: np.ndarray = np.load(tokenizer_classes_path)
        else:
            self.classes: np.ndarray = split_classes(
                tokens, batch_size, semantic_bert_path
            )
            print(f"Saving tokenizer classes at {tokenizer_classes_path}")
            np.save(tokenizer_classes_path, self.classes)

    def __str__(self):
        return "SemanticEntropyToken"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        logprobs = stats["greedy_log_probs"]
        sem_ent: List[List[float]] = []
        for s_lp in logprobs:
            sem_ent.append([])
            for lp in s_lp[:-1]:
                p = np.exp(lp)[: len(self.classes)]
                class_probs = np.bincount(self.classes, weights=p)
                sem_ent[-1].append(-np.mean(class_probs * np.log(class_probs)))
        return np.concatenate(sem_ent)


if __name__ == "__main__":
    print(
        split_classes(
            [
                "bad",
                "awful",
                "terrible",
                "horrible",
                "good",
                "fine",
                "excellent",
                "great",
            ],
            100,
        )
    )
