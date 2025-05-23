import numpy as np

from typing import Dict, List, Tuple

from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.deberta import Deberta
from lm_polygraph.stat_calculators.greedy_alternatives_nli import eval_nli_model


class SemanticClassesClaimToSamplesCalculator(StatCalculator):
    """
    Computes the NLI relationship between each claim and the corresponding sample texts.

    This calculator constructs NLI input pairs between each claim and each sample sentence,
    then uses a DeBERTa-based NLI model to predict entailment, contradiction, or neutral.

    The results are organized as a 3-level nested list structure:
        [n_samples][n_claims_in_sample][n_sample_texts]

    where each innermost list contains NLI labels for all sample sentences in relation to a claim.

    This serves as a dependency for estimators that assess claim strength or validity using
    context-based entailment signals.

    Required dependencies:
        - "sample_texts": List of sentences for each sample
        - "claims": List of Claim objects for each sample

    Provides:
        - "nli_to_sample_texts": List[List[List[Dict[str, float]]]] of NLI labels
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return ["nli_to_sample_texts"], ["sample_texts", "claims"]

    def __init__(self, nli_model: Deberta):
        """
        Initializes the calculator with an NLI model.

        Args:
            nli_model (Deberta): An instance of a DeBERTa-based NLI model wrapper.
        """

        super().__init__()
        self.nli_model = nli_model

    def __call__(
        self,
        dependencies: Dict[str, np.ndarray],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, List[List[List[Dict[str, float]]]]]:
        """
        Runs the NLI model over all claim-sentence pairs.

        Args:
            dependencies (Dict): Dictionary containing:
                - "sample_texts": List[List[str]] of sentences for each input sample.
                - "claims": List[List[Claim]] of Claim objects per sample.
            texts (List[str]): Not used, present for interface compatibility.
            model (WhiteboxModel): Not used, present for interface compatibility.
            max_new_tokens (int): Not used.

        Returns:
            Dict[str, List[List[List[Dict[str, float]]]]]: A mapping with key "nli_to_sample_texts",
            containing a nested list of NLI labels.
        """

        self.nli_model.setup()

        # Prepare NLI input pairs
        nli_inputs: List[Tuple[str, str]] = []
        for sample_claims, sample_sentences in zip(
            dependencies["claims"],
            dependencies["sample_texts"],
        ):
            for claim in sample_claims:
                nli_inputs.extend(
                    (sentence, claim.claim_text) for sentence in sample_sentences
                )

        # Evaluate all pairs using the NLI model
        nli_outputs: List[Dict[str, float]] = eval_nli_model(nli_inputs, self.nli_model)

        # Organize outputs into [sample][claim][sentence] structure
        structured_outputs: List[List[List[Dict[str, float]]]] = []
        idx = 0
        for sample_claims, sample_sentences in zip(
            dependencies["claims"],
            dependencies["sample_texts"],
        ):
            sample_result = []
            for _ in sample_claims:
                claim_result = nli_outputs[idx : idx + len(sample_sentences)]
                sample_result.append(claim_result)
                idx += len(sample_sentences)
            structured_outputs.append(sample_result)

        return {"nli_to_sample_texts": structured_outputs}
