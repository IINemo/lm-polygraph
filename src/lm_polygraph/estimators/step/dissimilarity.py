import numpy as np

from typing import Dict
from rouge_score import rouge_scorer
from lm_polygraph.estimators.estimator import Estimator
from lm_polygraph.stat_calculators.step.utils import flatten, reconstruct


class StepsDissimilarity(Estimator):
    def __init__(
            self,
            similarity: str,
    ):
        assert similarity in [
            "rouge1",
            "rouge2",
            "rougeL",
            "nli_entail",
            "nli_contra",
            "nli_ccp",
        ]
        self.similarity = similarity
        if similarity.startswith("nli_"):
            super().__init__(["steps_greedy_nli_similarity"], "claim")
        else:
            super().__init__(["sample_steps_texts", "claims"], "claim")

    def __str__(self):
        return "StepsDissimilarity " + self.similarity

    def ccp(self, e: float, c: float) -> float:
        return e / (e + c + 1e-9)

    def nli_similarities(self, stats: dict) -> list[list[float]]:
        nlis: list[dict] = flatten(stats["steps_greedy_nli_similarity"])
        fwd: list[list[float]] = []
        bwd: list[list[float]] = []
        for sample_nlis in nlis:
            fwd.append([])
            bwd.append([])
            for i in range(len(sample_nlis["forward_entailment"])):
                if self.similarity == "nli_entail":
                    fwd[-1].append(sample_nlis["forward_entailment"][i])
                    bwd[-1].append(sample_nlis["backward_entailment"][i])
                elif self.similarity == "nli_contra":
                    fwd[-1].append(1 - sample_nlis["forward_contradiction"][i])
                    bwd[-1].append(1 - sample_nlis["backward_contradiction"][i])
                elif self.similarity == "nli_ccp":
                    fwd[-1].append(self.ccp(
                        sample_nlis["forward_entailment"][i],
                        sample_nlis["forward_contradiction"][i],
                    ))
                    bwd[-1].append(self.ccp(
                        sample_nlis["backward_entailment"][i],
                        sample_nlis["backward_contradiction"][i],
                    ))
                else:
                    raise Exception(f"Unknown similarity: {self.similarity}")
        return [
            [(f + b) / 2 for f, b in zip(f_sc, b_sc)]
            for f_sc, b_sc in zip(fwd, bwd)
        ]

    def rouge_similarities(self, stats: dict) -> list[list[float]]:
        sample_texts: list[list[str]] = flatten(stats["sample_steps_texts"])
        greedy_texts: list[str] = [x.claim_text for x in flatten(stats["claims"])]
        assert len(sample_texts) == len(greedy_texts)
        scorer = rouge_scorer.RougeScorer([self.similarity], use_stemmer=True)
        sims = []
        for greedy_text, cur_sample_texts in zip(greedy_texts, sample_texts):
            sims.append([])
            for sample_text in cur_sample_texts:
                sims[-1].append(scorer.score(greedy_text, sample_text)[self.similarity].fmeasure)
        return sims

    def similarities(self, stats: dict) -> list[list[float]]:
        if self.similarity.startswith("nli_"):
            return self.nli_similarities(stats)
        elif self.similarity.startswith("rouge"):
            return self.rouge_similarities(stats)
        else:
            raise Exception(f"Unknown similarity: {self.similarity}")

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        sims: list[list[float]] = self.similarities(stats)
        dissim = [1 - np.mean(sim) for sim in sims]
        return reconstruct(dissim, stats["steps_greedy_nli_similarity"])
