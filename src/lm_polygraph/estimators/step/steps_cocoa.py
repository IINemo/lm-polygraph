import numpy as np
from typing import Dict

from ..estimator import Estimator


# TODO: estimate's name need to be changed --> StepsCocoaMSE
class StepsCocoaMTE(Estimator):
    """
    Step-wise version of CocoaMTE (Cocoa Mean Token Entropy) estimator.

    This estimator combines step-wise entropy with step-wise semantic similarity
    to provide enhanced uncertainty estimation at the step level.

    Supports multiple similarity calculators:
    - StepsGreedySimilarityCalculator (steps_greedy_sentence_similarity)
    - StepsCrossEncoderSimilarityCalculator (steps_sample_sentence_similarity)
    """

    def __init__(self, similarity_key: str = "steps_greedy_sentence_similarity"):
        """
        Initialize the estimator.

        Args:
            similarity_key: Key for similarity data in stats. Options:
                - "steps_greedy_sentence_similarity" (from StepsGreedySimilarityCalculator)
                - "steps_sample_sentence_similarity" (from StepsCrossEncoderSimilarityCalculator)
        """
        self.similarity_key = similarity_key
        super().__init__([similarity_key, "steps_entropy"], "sequence")

    def __str__(self):
        return f"StepsCocoaMTE({self.similarity_key})"

    def _normalize_similarity_data(self, similarity_data) -> list:
        """
        Normalize different similarity data formats to the standard CoCoA format.

        Expected output format: [batch_size][n_steps][n_samples]
        Where each element is a similarity score between greedy and sample.

        Args:
            similarity_data: Raw similarity data from different calculators

        Returns:
            Normalized similarity data in format [batch_size][n_steps][n_samples]
        """
        if self.similarity_key == "steps_greedy_sentence_similarity":
            # Format: [batch_size][n_steps][n_samples]
            # Already in the correct format
            return similarity_data

        elif self.similarity_key == "steps_sample_sentence_similarity":
            # Format: [batch_size][n_steps] where each element is (n_samples, n_samples) matrix
            # Need to extract greedy-to-sample similarities
            normalized_data = []

            for batch_item in similarity_data:  # Each batch
                batch_steps = []
                for step_matrix in batch_item:  # Each step's similarity matrix
                    # Assume the first row contains greedy-to-sample similarities
                    # Or take the diagonal if it's self-similarity
                    # For now, take the mean of the first row (excluding self-similarity)
                    if step_matrix.shape[0] > 1:
                        # Take first row (greedy vs all samples), excluding diagonal
                        greedy_similarities = step_matrix[0, :]
                        batch_steps.append(greedy_similarities.tolist())
                    else:
                        # Single sample case
                        batch_steps.append([step_matrix[0, 0]])
                normalized_data.append(batch_steps)

            return normalized_data

        else:
            raise ValueError(f"Unsupported similarity_key: {self.similarity_key}")

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate step-wise enhanced entropy using semantic similarity.

        Parameters:
            stats (Dict[str, np.ndarray]): Input statistics containing:
                - similarity_key: Similarity data (format depends on calculator used)
                - 'steps_entropy': Step-wise entropy values

        Returns:
            Same structure as StepsSemanticEntropy: Enhanced entropy scores for each step
        """
        raw_similarity_data = stats[self.similarity_key]
        batch_steps_entropy = stats["steps_entropy"]

        # Normalize similarity data to standard format
        batch_steps_greedy_sentence_similarity = self._normalize_similarity_data(
            raw_similarity_data
        )

        enriched_entropy = []

        for sample_steps_greedy_similarity, sample_steps_entropy in zip(
            batch_steps_greedy_sentence_similarity, batch_steps_entropy
        ):
            # For each step in this sample
            sample_enhanced_entropy = []

            for step_greedy_similarity, step_entropy in zip(
                sample_steps_greedy_similarity, sample_steps_entropy
            ):
                # Implement CoCoA formula: C_CoCoA = C_inf * C_cons
                # C_inf = step_avg_entropy (information-theoretic confidence)
                # C_cons = (1/M) * Σ(1 - s*i) where s*i are similarity scores

                # Calculate consistency term C_cons
                # step_greedy_similarity: [n_samples] - similarity scores for this step
                dissimilarities = 1 - np.array(step_greedy_similarity)  # (1 - s*i)
                c_cons = np.mean(dissimilarities)  # (1/M) * Σ(1 - s*i)

                # Compute average entropy for this step (information-theoretic confidence)
                step_avg_entropy = np.mean(step_entropy)

                # Apply CoCoA formula: enhanced entropy = entropy * consistency
                enhanced_step_entropy = step_avg_entropy * c_cons
                sample_enhanced_entropy.append(enhanced_step_entropy)

            # Keep the step-wise structure instead of averaging across steps
            enriched_entropy.append(np.array(sample_enhanced_entropy))

        return enriched_entropy


class StepsCocoaSEE(Estimator):
    """
    Step-wise version of CocoaSEE (Cocoa Semantic Entropy Estimator).

    This estimator combines step-wise semantic entropy with step-wise semantic similarity
    to provide enhanced uncertainty estimation at the step level.

    Supports multiple similarity calculators:
    - StepsGreedySimilarityCalculator (steps_greedy_sentence_similarity)
    - StepsCrossEncoderSimilarityCalculator (steps_sample_sentence_similarity)

    Note: This estimator requires the output from StepsSemanticEntropy estimator to be
    passed as a separate parameter, not as a statistic.
    """

    def __init__(self, similarity_key: str = "steps_greedy_sentence_similarity"):
        """
        Initialize the estimator.

        Args:
            similarity_key: Key for similarity data in stats. Options:
                - "steps_greedy_sentence_similarity" (from StepsGreedySimilarityCalculator)
                - "steps_sample_sentence_similarity" (from StepsCrossEncoderSimilarityCalculator)
        """
        self.similarity_key = similarity_key
        super().__init__([similarity_key], "sequence")

    def __str__(self):
        return f"StepsCocoaSEE({self.similarity_key})"

    def _normalize_similarity_data(self, similarity_data) -> list:
        """
        Normalize different similarity data formats to the standard CoCoA format.

        Expected output format: [batch_size][n_steps][n_samples]
        Where each element is a similarity score between greedy and sample.
        """
        if self.similarity_key == "steps_greedy_sentence_similarity":
            return similarity_data

        elif self.similarity_key == "steps_sample_sentence_similarity":
            normalized_data = []

            for batch_item in similarity_data:
                batch_steps = []
                for step_matrix in batch_item:
                    if step_matrix.shape[0] > 1:
                        greedy_similarities = step_matrix[0, :]
                        batch_steps.append(greedy_similarities.tolist())
                    else:
                        batch_steps.append([step_matrix[0, 0]])
                normalized_data.append(batch_steps)

            return normalized_data

        else:
            raise ValueError(f"Unsupported similarity_key: {self.similarity_key}")

    def __call__(
        self, stats: Dict[str, np.ndarray], semantic_entropy_output: np.ndarray = None
    ) -> np.ndarray:
        """
        Calculate step-wise enhanced semantic entropy using semantic similarity.

        Parameters:
            stats (Dict[str, np.ndarray]): Input statistics containing similarity data
            semantic_entropy_output: Output from StepsSemanticEntropy estimator

        Returns:
            Same structure as semantic_entropy_output: Enhanced entropy scores for each step
        """
        if semantic_entropy_output is None:
            raise ValueError(
                "semantic_entropy_output must be provided. This should be the output from StepsSemanticEntropy estimator."
            )

        raw_similarity_data = stats[self.similarity_key]
        batch_steps_semantic_entropy = semantic_entropy_output

        # Normalize similarity data to standard format
        batch_steps_greedy_sentence_similarity = self._normalize_similarity_data(
            raw_similarity_data
        )

        # Initialize result with same structure as input
        enriched_semantic_entropy = []

        for sample_steps_greedy_similarity, sample_steps_semantic_entropy in zip(
            batch_steps_greedy_sentence_similarity, batch_steps_semantic_entropy
        ):
            # For each step in this sample, keep the step-wise structure
            sample_enhanced_semantic_entropy = []

            for step_greedy_similarity, step_semantic_entropy in zip(
                sample_steps_greedy_similarity, sample_steps_semantic_entropy
            ):
                # Implement CoCoA formula: C_CoCoA = C_inf * C_cons
                # C_inf = step_semantic_entropy (information-theoretic confidence)
                # C_cons = (1/M) * Σ(1 - s*i) where s*i are similarity scores

                # Calculate consistency term C_cons
                dissimilarities = 1 - np.array(step_greedy_similarity)
                c_cons = np.mean(dissimilarities)

                # Apply CoCoA formula: enhanced semantic entropy = semantic_entropy * consistency
                enhanced_step_semantic_entropy = step_semantic_entropy * c_cons
                sample_enhanced_semantic_entropy.append(enhanced_step_semantic_entropy)

            # Convert to numpy array to match StepsSemanticEntropy output format
            enriched_semantic_entropy.append(np.array(sample_enhanced_semantic_entropy))

        return enriched_semantic_entropy


class StepsCocoaMSP(Estimator):
    """
    Step-wise version of CocoaMSP (Cocoa Maximum Sequence Probability) estimator.

    This estimator combines step-wise log likelihoods with step-wise semantic similarity
    to provide enhanced uncertainty estimation at the step level.

    Supports multiple similarity calculators.
    """

    def __init__(self, similarity_key: str = "steps_greedy_sentence_similarity"):
        """
        Initialize the estimator.

        Args:
            similarity_key: Key for similarity data in stats.
        """
        self.similarity_key = similarity_key
        super().__init__([similarity_key, "sample_steps_log_likelihoods"], "sequence")

    def __str__(self):
        return f"StepsCocoaMSP({self.similarity_key})"

    def _normalize_similarity_data(self, similarity_data) -> list:
        """Normalize similarity data to standard format."""
        if self.similarity_key == "steps_greedy_sentence_similarity":
            return similarity_data

        elif self.similarity_key == "steps_sample_sentence_similarity":
            normalized_data = []

            for batch_item in similarity_data:
                batch_steps = []
                for step_matrix in batch_item:
                    if step_matrix.shape[0] > 1:
                        greedy_similarities = step_matrix[0, :]
                        batch_steps.append(greedy_similarities.tolist())
                    else:
                        batch_steps.append([step_matrix[0, 0]])
                normalized_data.append(batch_steps)

            return normalized_data

        else:
            raise ValueError(f"Unsupported similarity_key: {self.similarity_key}")

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate step-wise enhanced log likelihoods using semantic similarity.
        """
        raw_similarity_data = stats[self.similarity_key]
        batch_steps_log_likelihoods = stats["sample_steps_log_likelihoods"]

        # Normalize similarity data
        batch_steps_greedy_sentence_similarity = self._normalize_similarity_data(
            raw_similarity_data
        )

        enriched_metrics = []

        for sample_steps_greedy_similarity, sample_steps_log_likelihoods in zip(
            batch_steps_greedy_sentence_similarity, batch_steps_log_likelihoods
        ):
            sample_enhanced_metrics = []

            for step_greedy_similarity, step_log_likelihoods in zip(
                sample_steps_greedy_similarity, sample_steps_log_likelihoods
            ):
                # Compute average dissimilarity for this step
                step_dissimilarity = 1 - np.array(step_greedy_similarity)
                avg_dissimilarity = np.mean(step_dissimilarity)

                # Compute sum of log likelihoods for this step
                if len(step_log_likelihoods) == 0:
                    step_sum_ll = 0.0
                elif isinstance(step_log_likelihoods[0], (list, np.ndarray)):
                    all_step_lls = []
                    for sample_ll in step_log_likelihoods:
                        all_step_lls.extend(sample_ll)
                    step_sum_ll = np.sum(all_step_lls) if all_step_lls else 0.0
                else:
                    step_sum_ll = np.sum(step_log_likelihoods)

                # Enhanced metric = -sum_ll * avg_dissimilarity
                enhanced_step_metric = -step_sum_ll * avg_dissimilarity
                sample_enhanced_metrics.append(enhanced_step_metric)

            enriched_metrics.append(np.array(sample_enhanced_metrics))

        return enriched_metrics


class StepsCocoaPPL(Estimator):
    """
    Step-wise version of CocoaPPL (Cocoa Perplexity) estimator.

    This estimator combines step-wise perplexity with step-wise semantic similarity
    to provide enhanced uncertainty estimation at the step level.

    Supports multiple similarity calculators.
    """

    def __init__(self, similarity_key: str = "steps_greedy_sentence_similarity"):
        """
        Initialize the estimator.

        Args:
            similarity_key: Key for similarity data in stats.
        """
        self.similarity_key = similarity_key
        super().__init__([similarity_key, "sample_steps_log_likelihoods"], "sequence")

    def __str__(self):
        return f"StepsCocoaPPL({self.similarity_key})"

    def _normalize_similarity_data(self, similarity_data) -> list:
        """Normalize similarity data to standard format."""
        if self.similarity_key == "steps_greedy_sentence_similarity":
            return similarity_data

        elif self.similarity_key == "steps_sample_sentence_similarity":
            normalized_data = []

            for batch_item in similarity_data:
                batch_steps = []
                for step_matrix in batch_item:
                    if step_matrix.shape[0] > 1:
                        greedy_similarities = step_matrix[0, :]
                        batch_steps.append(greedy_similarities.tolist())
                    else:
                        batch_steps.append([step_matrix[0, 0]])
                normalized_data.append(batch_steps)

            return normalized_data

        else:
            raise ValueError(f"Unsupported similarity_key: {self.similarity_key}")

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate step-wise enhanced perplexity using semantic similarity.
        """
        raw_similarity_data = stats[self.similarity_key]
        batch_steps_log_likelihoods = stats["sample_steps_log_likelihoods"]

        # Normalize similarity data
        batch_steps_greedy_sentence_similarity = self._normalize_similarity_data(
            raw_similarity_data
        )

        enriched_ppl = []

        for sample_steps_greedy_similarity, sample_steps_log_likelihoods in zip(
            batch_steps_greedy_sentence_similarity, batch_steps_log_likelihoods
        ):
            sample_enhanced_ppl = []

            for step_greedy_similarity, step_log_likelihoods in zip(
                sample_steps_greedy_similarity, sample_steps_log_likelihoods
            ):
                # Compute average dissimilarity for this step
                step_dissimilarity = 1 - np.array(step_greedy_similarity)
                avg_dissimilarity = np.mean(step_dissimilarity)

                # Compute perplexity for this step
                if len(step_log_likelihoods) == 0:
                    step_ppl = 0.0
                elif isinstance(step_log_likelihoods[0], (list, np.ndarray)):
                    all_step_lls = []
                    for sample_ll in step_log_likelihoods:
                        all_step_lls.extend(sample_ll)
                    step_ppl = -np.mean(all_step_lls) if all_step_lls else 0.0
                else:
                    step_ppl = -np.mean(step_log_likelihoods)

                # Enhanced perplexity = ppl * avg_dissimilarity
                enhanced_step_ppl = step_ppl * avg_dissimilarity
                sample_enhanced_ppl.append(enhanced_step_ppl)

            enriched_ppl.append(np.array(sample_enhanced_ppl))

        return enriched_ppl
