import numpy as np

from typing import Dict

from .estimator import Estimator

class Prompt_Coh(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "P(True)" as provided in the paper https://arxiv.org/abs/2207.05221.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    On input question q and model generation a, the method uses the following prompt:
        Question: {q}
        Possible answer: {a}
        Is the possible answer:
         (A) True
         (B) False
        The possible answer is:
    and calculates the log-probability of 'True' token with minus sign.
    """

    def __init__(self):
        super().__init__(["prompt_coherence"], "sequence")

    def __str__(self):
        return "prompt_coherence"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates minus log-probability of true token for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * probabilities of the true token in 'p_true'
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        ptrue = stats["prompt_coherence"]
        return -np.array(ptrue)
    
class Prompt_Con(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "P(True)" as provided in the paper https://arxiv.org/abs/2207.05221.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    On input question q and model generation a, the method uses the following prompt:
        Question: {q}
        Possible answer: {a}
        Is the possible answer:
         (A) True
         (B) False
        The possible answer is:
    and calculates the log-probability of 'True' token with minus sign.
    """

    def __init__(self):
        super().__init__(["prompt_consistency"], "sequence")

    def __str__(self):
        return "prompt_consistency"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates minus log-probability of true token for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * probabilities of the true token in 'p_true'
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        ptrue = stats["prompt_consistency"]
        return -np.array(ptrue)
    
class Prompt_Rel(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "P(True)" as provided in the paper https://arxiv.org/abs/2207.05221.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    On input question q and model generation a, the method uses the following prompt:
        Question: {q}
        Possible answer: {a}
        Is the possible answer:
         (A) True
         (B) False
        The possible answer is:
    and calculates the log-probability of 'True' token with minus sign.
    """

    def __init__(self):
        super().__init__(["prompt_relevance"], "sequence")

    def __str__(self):
        return "prompt_relevance"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates minus log-probability of true token for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * probabilities of the true token in 'p_true'
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        ptrue = stats["prompt_relevance"]
        return -np.array(ptrue)
    
class Prompt_Flu(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "P(True)" as provided in the paper https://arxiv.org/abs/2207.05221.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    On input question q and model generation a, the method uses the following prompt:
        Question: {q}
        Possible answer: {a}
        Is the possible answer:
         (A) True
         (B) False
        The possible answer is:
    and calculates the log-probability of 'True' token with minus sign.
    """

    def __init__(self):
        super().__init__(["prompt_fluency"], "sequence")

    def __str__(self):
        return "prompt_fluency"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates minus log-probability of true token for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * probabilities of the true token in 'p_true'
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        ptrue = stats["prompt_fluency"]
        return -np.array(ptrue)
    
class Prompt_Sum(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "P(True)" as provided in the paper https://arxiv.org/abs/2207.05221.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    On input question q and model generation a, the method uses the following prompt:
        Question: {q}
        Possible answer: {a}
        Is the possible answer:
         (A) True
         (B) False
        The possible answer is:
    and calculates the log-probability of 'True' token with minus sign.
    """

    def __init__(self):
        super().__init__(["prompt_coherence", "prompt_consistency", "prompt_relevance", "prompt_fluency"], "sequence")

    def __str__(self):
        return "Prompt_Sum"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates minus log-probability of true token for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * probabilities of the true token in 'p_true'
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        ptrue = 0
        for s in ["prompt_coherence", "prompt_consistency", "prompt_relevance", "prompt_fluency"]:
            ptrue += stats[s]
        return -np.array(ptrue)




class Prompt_UE(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "P(True)" as provided in the paper https://arxiv.org/abs/2207.05221.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    On input question q and model generation a, the method uses the following prompt:
        Question: {q}
        Possible answer: {a}
        Is the possible answer:
         (A) True
         (B) False
        The possible answer is:
    and calculates the log-probability of 'True' token with minus sign.
    """

    def __init__(self):
        super().__init__(["prompt_ue"], "sequence")

    def __str__(self):
        return "prompt_ue"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates minus log-probability of true token for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * probabilities of the true token in 'p_true'
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        ptrue = stats["prompt_ue"]
        return np.array(ptrue)
    
class Prompt_Quality(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "P(True)" as provided in the paper https://arxiv.org/abs/2207.05221.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    On input question q and model generation a, the method uses the following prompt:
        Question: {q}
        Possible answer: {a}
        Is the possible answer:
         (A) True
         (B) False
        The possible answer is:
    and calculates the log-probability of 'True' token with minus sign.
    """

    def __init__(self):
        super().__init__(["prompt_quality"], "sequence")

    def __str__(self):
        return "prompt_quality"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates minus log-probability of true token for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * probabilities of the true token in 'p_true'
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        ptrue = stats["prompt_quality"]
        return -np.array(ptrue)
