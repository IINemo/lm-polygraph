import numpy as np
import itertools
from typing import Dict, List, Tuple

from ..stat_calculator import StatCalculator
from sentence_transformers import CrossEncoder
from lm_polygraph.utils.model import WhiteboxModel


class StepsCrossEncoderSimilarityCalculator(StatCalculator):
    """
    Calculates step-wise cross-encoder similarity matrices for generation samples using RoBERTa model.
    
    For each reasoning step, computes:
    1. sample_sentence_similarity: similarity matrix between samples at each step
    2. sample_token_similarity: token-level similarities for samples at each step  
    3. token_similarity: token-level similarities for greedy generation at each step
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """
        return [
            "steps_sample_sentence_similarity",
            "steps_sample_token_similarity", 
            "steps_token_similarity",
        ], ["input_texts", "sample_steps_tokens", "sample_steps_texts", "claims", "greedy_tokens"]

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

    def _extract_text_content(self, input_data) -> str:
        """
        Extract text content from various input formats.
        
        Args:
            input_data: Can be:
                - String: return as is
                - List of dicts with 'content' key: extract content
                - Other: convert to string
        """
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, list) and len(input_data) > 0:
            if isinstance(input_data[0], dict) and 'content' in input_data[0]:
                # Extract content from the first message (assuming user input)
                return input_data[0]['content']
            else:
                # Fallback: join all items as strings
                return " ".join([str(item) for item in input_data])
        else:
            return str(input_data) if input_data is not None else ""

    def _build_cumulative_text(self, input_text: str, steps: List[str], current_step_idx: int) -> str:
        """
        Build cumulative text: input + step1 + step2 + ... + current_step
        """
        cumulative_steps = steps[:current_step_idx + 1]
        return input_text + " " + " ".join(cumulative_steps)

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        device = model.device()
        tokenizer = model.tokenizer

        if not self.crossencoder_setup:
            self._setup(device=device)
            self.crossencoder_setup = True

        batch_sample_steps_tokens = dependencies["sample_steps_tokens"]
        batch_sample_steps_texts = dependencies["sample_steps_texts"] 
        batch_input_texts = dependencies["input_texts"]
        batch_claims = dependencies["claims"]  # greedy reasoning steps
        batch_greedy_tokens = dependencies["greedy_tokens"]

        special_tokens = list(model.tokenizer.added_tokens_decoder.keys())

        # Results containers
        batch_steps_sample_sentence_similarity = []
        batch_steps_sample_token_similarity = []
        batch_steps_token_similarity = []

        # Process each sample in the batch
        for sample_idx, (input_text, sample_steps_texts, sample_steps_tokens, claims, greedy_tokens) in enumerate(
            zip(batch_input_texts, batch_sample_steps_texts, batch_sample_steps_tokens, batch_claims, batch_greedy_tokens)
        ):
            n_steps = len(sample_steps_texts)
            
            # Initialize containers for this sample
            steps_sample_sentence_similarity = []
            steps_sample_token_similarity = []
            steps_token_similarity = []

            # Process each step
            for step_idx in range(n_steps):
                # 1. Process sample sentence similarity for this step
                step_texts = sample_steps_texts[step_idx]  # All sample texts for this step
                
                # Build cumulative texts for samples
                cumulative_sample_texts = []
                for sample_text in step_texts:
                    # For samples: input + previous_steps + current_sample_step
                    # Extract claim texts from Claim objects
                    if isinstance(claims, list) and len(claims) > 0:
                        # Handle Claim objects with claim_text attribute
                        claims_str = []
                        for claim in claims:
                            if hasattr(claim, 'claim_text'):
                                claims_str.append(claim.claim_text)
                            else:
                                claims_str.append(str(claim))
                        prev_steps = claims_str[:step_idx] if step_idx > 0 else []
                    else:
                        prev_steps = []
                    
                    # Extract text content from input_text
                    input_text_str = self._extract_text_content(input_text)
                    cumulative_text = input_text_str + " " + " ".join(prev_steps + [str(sample_text)])
                    cumulative_sample_texts.append(cumulative_text)

                # Compute similarity matrix for this step's samples (same logic as original)
                unique_texts, inv = np.unique(cumulative_sample_texts, return_inverse=True)
                pairs = list(itertools.product(unique_texts, unique_texts))
                
                if len(pairs) > 0:
                    sim_scores = self.crossencoder.predict(pairs, batch_size=self.batch_size)
                    unique_mat_shape = (len(unique_texts), len(unique_texts))
                    sim_scores_matrix = sim_scores.reshape(unique_mat_shape)
                    # Recover full matrix
                    step_sim_matrix = sim_scores_matrix[inv, :][:, inv]
                else:
                    step_sim_matrix = np.array([[1.0]])  # Single sample case
                
                steps_sample_sentence_similarity.append(step_sim_matrix)

                # 2. Process sample token similarity for this step
                step_tokens = sample_steps_tokens[step_idx]  # All sample tokens for this step
                step_samples_token_scores = []
                
                for sample_tokens in step_tokens:
                    if len(sample_tokens) > 1:
                        is_special_tokens = np.isin(sample_tokens, special_tokens)
                        cropped_tokens = list(itertools.combinations(sample_tokens, len(sample_tokens) - 1))[::-1]
                        
                        # Build cumulative text for this sample
                        # Extract claim texts from Claim objects
                        if isinstance(claims, list) and len(claims) > 0:
                            claims_str = []
                            for claim in claims:
                                if hasattr(claim, 'claim_text'):
                                    claims_str.append(claim.claim_text)
                                else:
                                    claims_str.append(str(claim))
                            prev_steps = claims_str[:step_idx] if step_idx > 0 else []
                        else:
                            prev_steps = []
                        # Extract text content from input_text
                        input_text_str = self._extract_text_content(input_text)
                        base_cumulative = input_text_str + " " + " ".join(prev_steps)
                        
                        # Full text with current step
                        raw_text = base_cumulative + " " + tokenizer.decode(sample_tokens, skip_special_tokens=True)
                        
                        # Create batches comparing full vs cropped
                        batches = [
                            (
                                raw_text,
                                base_cumulative + " " + tokenizer.decode(list(t), skip_special_tokens=True),
                            )
                            for t in cropped_tokens
                        ]
                        
                        token_scores = self.crossencoder.predict(batches, batch_size=self.batch_size)
                        token_scores[is_special_tokens] = 1
                    else:
                        token_scores = np.array([0.5] * len(sample_tokens))
                    
                    step_samples_token_scores.append(token_scores)
                
                steps_sample_token_similarity.append(step_samples_token_scores)

                # 3. Process greedy token similarity for this step
                if step_idx < len(claims):
                    # Get tokens for this greedy step
                    # Note: This is a simplification - we'd need actual step-wise greedy tokens
                    # For now, we'll approximate by using a portion of greedy_tokens
                    step_start = step_idx * (len(greedy_tokens) // n_steps) if n_steps > 0 else 0
                    step_end = min((step_idx + 1) * (len(greedy_tokens) // n_steps), len(greedy_tokens))
                    step_greedy_tokens = greedy_tokens[step_start:step_end]
                    
                    if len(step_greedy_tokens) > 1:
                        is_special_tokens = np.isin(step_greedy_tokens, special_tokens)
                        cropped_tokens = list(itertools.combinations(step_greedy_tokens, len(step_greedy_tokens) - 1))[::-1]
                        
                        # Build cumulative text for greedy
                        # Extract claim texts from Claim objects
                        if isinstance(claims, list) and len(claims) > 0:
                            claims_str = []
                            for claim in claims:
                                if hasattr(claim, 'claim_text'):
                                    claims_str.append(claim.claim_text)
                                else:
                                    claims_str.append(str(claim))
                            prev_steps = claims_str[:step_idx] if step_idx > 0 else []
                        else:
                            prev_steps = []
                        # Extract text content from input_text
                        input_text_str = self._extract_text_content(input_text)
                        base_cumulative = input_text_str + " " + " ".join(prev_steps)
                        
                        raw_text = base_cumulative + " " + tokenizer.decode(step_greedy_tokens, skip_special_tokens=True)
                        
                        batches = [
                            (
                                raw_text,
                                base_cumulative + " " + tokenizer.decode(list(t), skip_special_tokens=True),
                            )
                            for t in cropped_tokens
                        ]
                        
                        token_scores = self.crossencoder.predict(batches, batch_size=self.batch_size)
                        token_scores[is_special_tokens] = 1
                    else:
                        token_scores = np.array([0.5] * len(step_greedy_tokens)) if len(step_greedy_tokens) > 0 else np.array([0.5])
                    
                    steps_token_similarity.append(token_scores)
                else:
                    steps_token_similarity.append(np.array([0.5]))  # Fallback


            
            # Add this sample's results to batch results
            batch_steps_sample_sentence_similarity.append(steps_sample_sentence_similarity)
            batch_steps_sample_token_similarity.append(steps_sample_token_similarity)
            batch_steps_token_similarity.append(steps_token_similarity)

        return {
            "steps_sample_sentence_similarity": batch_steps_sample_sentence_similarity,
            "steps_sample_token_similarity": batch_steps_sample_token_similarity,
            "steps_token_similarity": batch_steps_token_similarity,
        } 