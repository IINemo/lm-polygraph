import numpy as np
from typing import Dict

from .estimator import Estimator


class StepsCocoaMTE(Estimator):
    """
    Step-wise version of CocoaMTE (Cocoa Maximum Token Entropy) estimator.
    
    This estimator combines step-wise entropy with step-wise semantic similarity
    to provide enhanced uncertainty estimation at the step level.
    """
    
    def __init__(self):
        super().__init__(["steps_greedy_sentence_similarity", "steps_entropy"], "sequence")

    def __str__(self):
        return "StepsCocoaMTE"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate step-wise enhanced entropy using semantic similarity.
        
        Parameters:
            stats (Dict[str, np.ndarray]): Input statistics containing:
                - 'steps_greedy_sentence_similarity' (List[List[List[float]]]): 
                    Shape: [batch_size][n_steps][n_samples]
                    For each sample, for each step: similarity scores between greedy and sample steps
                - 'steps_entropy' (List[List[List[float]]]): 
                    For each sample, for each step: entropy values for each sample
                    
        Returns:
            np.ndarray: Enhanced entropy scores for each sample
        """
        batch_steps_greedy_sentence_similarity = stats["steps_greedy_sentence_similarity"]
        batch_steps_entropy = stats["steps_entropy"]

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
            
            # Average across all steps for this sample
            sample_final_entropy = np.mean(sample_enhanced_entropy)
            enriched_entropy.append(sample_final_entropy)

        return np.array(enriched_entropy)


class StepsCocoaSEE(Estimator):
    """
    Step-wise version of CocoaSEE (Cocoa Semantic Entropy Estimator).
    
    This estimator combines step-wise semantic entropy with step-wise semantic similarity
    to provide enhanced uncertainty estimation at the step level.
    
    Note: This estimator requires the output from StepsSemanticEntropy estimator to be
    passed as a separate parameter, not as a statistic.
    """
    
    def __init__(self):
        super().__init__(["steps_greedy_sentence_similarity"], "sequence")

    def __str__(self):
        return "StepsCocoaSEE"

    def __call__(self, stats: Dict[str, np.ndarray], semantic_entropy_output: np.ndarray = None) -> np.ndarray:
        """
        Calculate step-wise enhanced semantic entropy using semantic similarity.
        
        Parameters:
            stats (Dict[str, np.ndarray]): Input statistics containing:
                - 'steps_greedy_sentence_similarity' (List[List[List[float]]]): 
                    Shape: [batch_size][n_steps][n_samples]
                    For each sample, for each step: similarity scores between greedy and sample steps
            semantic_entropy_output: Output from StepsSemanticEntropy estimator
                Structure: [array([step1, step2, ...]), array([step1, step2, ...]), ...]
                    
        Returns:
            Same structure as semantic_entropy_output: Enhanced semantic entropy scores
        """
        if semantic_entropy_output is None:
            raise ValueError("semantic_entropy_output must be provided. This should be the output from StepsSemanticEntropy estimator.")
        
        batch_steps_greedy_sentence_similarity = stats["steps_greedy_sentence_similarity"]
        batch_steps_semantic_entropy = semantic_entropy_output

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
                # step_greedy_similarity: [n_samples] - similarity scores for this step
                dissimilarities = 1 - np.array(step_greedy_similarity)  # (1 - s*i)
                c_cons = np.mean(dissimilarities)  # (1/M) * Σ(1 - s*i)
                
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
    """
    
    def __init__(self):
        super().__init__(["steps_greedy_sentence_similarity", "steps_log_likelihoods"], "sequence")

    def __str__(self):
        return "StepsCocoaMSP"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate step-wise enhanced log likelihoods using semantic similarity.
        
        Parameters:
            stats (Dict[str, np.ndarray]): Input statistics containing:
                - 'steps_greedy_sentence_similarity' (List[List[List[float]]]): 
                    For each sample, for each step: similarity scores between greedy and sample steps
                - 'steps_log_likelihoods' (List[List[List[float]]]): 
                    For each sample, for each step: log likelihood values for each sample
                    
        Returns:
            np.ndarray: Enhanced log likelihood scores for each sample
        """
        batch_steps_greedy_sentence_similarity = stats["steps_greedy_sentence_similarity"]
        batch_steps_log_likelihoods = stats["steps_log_likelihoods"]

        enriched_metrics = []

        for sample_steps_greedy_similarity, sample_steps_log_likelihoods in zip(
            batch_steps_greedy_sentence_similarity, batch_steps_log_likelihoods
        ):
            # For each step in this sample
            sample_enhanced_metrics = []
            
            for step_greedy_similarity, step_log_likelihoods in zip(
                sample_steps_greedy_similarity, sample_steps_log_likelihoods
            ):
                # Compute average dissimilarity for this step
                step_dissimilarity = 1 - np.array(step_greedy_similarity)
                avg_dissimilarity = np.mean(step_dissimilarity)
                
                # Compute sum of log likelihoods for this step
                step_sum_ll = np.sum(step_log_likelihoods)
                
                # Enhanced metric = -sum_ll * avg_dissimilarity
                enhanced_step_metric = -step_sum_ll * avg_dissimilarity
                sample_enhanced_metrics.append(enhanced_step_metric)
            
            # Average across all steps for this sample
            sample_final_metric = np.mean(sample_enhanced_metrics)
            enriched_metrics.append(sample_final_metric)

        return np.array(enriched_metrics)


class StepsCocoaPPL(Estimator):
    """
    Step-wise version of CocoaPPL (Cocoa Perplexity) estimator.
    
    This estimator combines step-wise perplexity with step-wise semantic similarity
    to provide enhanced uncertainty estimation at the step level.
    """
    
    def __init__(self):
        super().__init__(["steps_greedy_sentence_similarity", "steps_log_likelihoods"], "sequence")

    def __str__(self):
        return "StepsCocoaPPL"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate step-wise enhanced perplexity using semantic similarity.
        
        Parameters:
            stats (Dict[str, np.ndarray]): Input statistics containing:
                - 'steps_greedy_sentence_similarity' (List[List[List[float]]]): 
                    For each sample, for each step: similarity scores between greedy and sample steps
                - 'steps_log_likelihoods' (List[List[List[float]]]): 
                    For each sample, for each step: log likelihood values for each sample
                    
        Returns:
            np.ndarray: Enhanced perplexity scores for each sample
        """
        batch_steps_greedy_sentence_similarity = stats["steps_greedy_sentence_similarity"]
        batch_steps_log_likelihoods = stats["steps_log_likelihoods"]

        enriched_ppl = []

        for sample_steps_greedy_similarity, sample_steps_log_likelihoods in zip(
            batch_steps_greedy_sentence_similarity, batch_steps_log_likelihoods
        ):
            # For each step in this sample
            sample_enhanced_ppl = []
            
            for step_greedy_similarity, step_log_likelihoods in zip(
                sample_steps_greedy_similarity, sample_steps_log_likelihoods
            ):
                # Compute average dissimilarity for this step
                step_dissimilarity = 1 - np.array(step_greedy_similarity)
                avg_dissimilarity = np.mean(step_dissimilarity)
                
                # Compute perplexity for this step
                step_ppl = -np.mean(step_log_likelihoods)
                
                # Enhanced perplexity = ppl * avg_dissimilarity
                enhanced_step_ppl = step_ppl * avg_dissimilarity
                sample_enhanced_ppl.append(enhanced_step_ppl)
            
            # Average across all steps for this sample
            sample_final_ppl = np.mean(sample_enhanced_ppl)
            enriched_ppl.append(sample_final_ppl)

        return np.array(enriched_ppl) 