import numpy as np
import itertools
from typing import Dict, List, Tuple

from ..stat_calculator import StatCalculator
from sentence_transformers import CrossEncoder
from lm_polygraph.utils.model import WhiteboxModel

from .utils import flatten, reconstruct


class StepsGreedySimilarityCalculator(StatCalculator):
    """
    Calculates the cross-encoder similarity between greedy steps and sample steps using RoBERTa model.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return [
            "steps_greedy_sentence_similarity",
        ], ["sample_steps_texts", "claims", "greedy_texts"]

    def __init__(
        self,
        batch_size: int = 10,
        cross_encoder_name: str = "cross-encoder/stsb-roberta-large",
    ):
        super().__init__()
        self.crossencoder_setup = False
        self.batch_size = batch_size
        self.cross_encoder_name = cross_encoder_name

    def _setup(self, device="cuda"):
        self.crossencoder = CrossEncoder(self.cross_encoder_name, device=device)

    def parse_steps(self, x: str) -> str:
        import re
        x = re.sub(r'- Step \d+:\s*', '', x)
        x = x.replace('<Answer>:', 'Answer:')
        return x

    def parse_problem(self, x: str) -> str:
        return x.split('<Question>: ', 1)[-1].split('<|im_end|>', 1)[0].replace('  ', ' ').strip()

    def parse_solution(self, x: str) -> str:
        x = x.split('Reasoning Steps:\n')[-1].strip().replace('\n', ' ')
        return self.parse_steps(x)

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the cross-encoder similarity between greedy steps and sample steps using RoBERTa model.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, containing:
                - 'sample_steps_texts' (List[List[str]]): several sampling generations
                    for each input text in the batch.
                - 'claims' (List[List[Dict]]): claim information for each sample.
                - 'greedy_texts' (List[str]): greedy generation texts.
            texts (List[str]): Input texts batch used for model generation.
            model (WhiteboxModel): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - 'steps_greedy_sentence_similarity' (List[List[float]]): for each input text and step: 
                    similarity scores between greedy and sample steps.
        """

        device = model.device()

        if not self.crossencoder_setup:
            self._setup(device=device)
            self.crossencoder_setup = True

        sample_steps_texts = dependencies["sample_steps_texts"]
        batch_texts: list[list[str]] = flatten(sample_steps_texts)  # batch_texts[step_idx][alternative_idx]
        greedy_texts: list[str] = [x.claim_text for x in flatten(dependencies["claims"])]
        greedy_solutions: list[str] = [
            dependencies['greedy_texts'][i]
            for i in range(len(sample_steps_texts))
            for _ in sample_steps_texts[i]
        ]
        input_texts: list[str] = [texts[i] for i in range(len(sample_steps_texts)) for _ in sample_steps_texts[i]]
        assert len(batch_texts) == len(greedy_texts) == len(input_texts)

        batch_pairs = []
        for sample_texts, greedy_text, greedy_solution, input_text in zip(
                batch_texts,
                greedy_texts,
                greedy_solutions,
                input_texts,
        ):
            # Parse the steps
            greedy_step = self.parse_steps(greedy_text)
            sample_steps = [self.parse_steps(x) for x in sample_texts]
            
            # Create pairs for cross-encoder comparison
            # Compare greedy step with each sample step
            for sample_step in sample_steps:
                batch_pairs.append((greedy_step, sample_step))

        # Get similarity scores using cross-encoder
        if batch_pairs:
            sim_scores = self.crossencoder.predict(batch_pairs, batch_size=self.batch_size)
        else:
            sim_scores = []

        # Reconstruct the results to match the original structure
        # Result should be [batch_size][n_steps][n_samples]
        steps_greedy_sentence_similarity = []
        score_idx = 0
        
        for sample_texts in sample_steps_texts:  # Each sample
            sample_step_similarities = []
            for step_texts in sample_texts:  # Each step in this sample
                step_similarities = []
                for _ in step_texts:  # Each sample text in this step
                    if score_idx < len(sim_scores):
                        step_similarities.append(sim_scores[score_idx])
                        score_idx += 1
                    else:
                        step_similarities.append(0.0)  # fallback value
                sample_step_similarities.append(step_similarities)
            steps_greedy_sentence_similarity.append(sample_step_similarities)

        return {
            "steps_greedy_sentence_similarity": steps_greedy_sentence_similarity
        } 