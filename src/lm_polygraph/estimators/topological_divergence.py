import warnings
from typing import Dict, List, Tuple, Literal
from collections import defaultdict
import numpy as np
import torch
try:
    from joblib import Parallel, delayed
    IS_PARALLEL_AVAILABLE = True
except ImportError:
    IS_PARALLEL_AVAILABLE = False
    warnings.warn(
        "Joblib is not installed. Parallel processing for MTopDivCalculator will not be available. "
        "Please install it via 'pip install joblib' if you want to use parallel processing."
        )    
try:
    import ripserplusplus as rpp_py
except ImportError:
    raise ImportError(
    "Please install the 'ripserplusplus' package to use TopologicalDivergence estimator. "
    "You can install it via 'pip install ripserplusplus'."
    )
try:
    import mtd.barcodes as mtd
except ImportError:
    raise ImportError(
    "Please install the 'mtd' package to use TopologicalDivergence estimator.\n"
    "Installation steps:\n"
    "  1. git clone https://github.com/IlyaTrofimov/MTopDiv.git\n"
    "  2. cd MTopDiv && python setup.py install"
    )

from .estimator import Estimator

def transform_attention_scores_to_distances(
    attention_weights: torch.Tensor,
    zero_out: Literal["prompt", "response"],
    len_answer: int,
    lower_bound: float = 0.0,
) -> torch.Tensor:
    """Transform attention matrix to the matrix of distances between tokens.

    Parameters
    ----------
    attention_weights : torch.Tensor
        Attention matrixes of one sample (n_heads x n_tokens x n_tokens).
    zero_out : Literal['prompt', 'response']
        Determines whether to zero out distances between prompt tokens or response tokens.
    len_answer : int
        Length of the response.

    Returns
    -------
    torch.Tensor
        Distance matrix.

    """
    n_tokens = attention_weights.shape[1]
    distance_mx = 1 - torch.clamp(
        attention_weights, min=lower_bound
    )  # torch.where(attn_mx > lower_bound, attn_mx, 0.0)
    zero_diag = torch.ones(n_tokens, n_tokens) - torch.eye(n_tokens)
    distance_mx *= zero_diag.to(attention_weights.device).expand_as(
        distance_mx
    )  # torch.diag(torch.diag(distance_mx))
    distance_mx = torch.minimum(distance_mx.transpose(1, 2), distance_mx)

    if zero_out == "prompt":
        len_prompt = n_tokens - len_answer
        distance_mx[:, :len_prompt, :len_prompt] = 0
    elif zero_out == "response":
        distance_mx[:, -len_answer:, -len_answer:] = 0
    else:
        raise ValueError(f"Unsupported zero_out parameter: {zero_out}")

    return distance_mx.cpu().numpy()

def transform_distances_to_mtopdiv(distance_mx: np.ndarray) -> float:
    """
    Calculate MTopDiv value for the given attention matrix.
    """
    barcodes = rpp_py.run("--format distance --dim 1", distance_mx)
    barcodes = mtd.barc2array(barcodes)
    mtopdiv = mtd.get_score(barcodes, 0, "sum_length")
    return mtopdiv


class TopologicalDivergence(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of 
    "Hallucination Detection in LLMs with Topological Divergence on Attention Graphs" 
    as provided in the paper https://arxiv.org/abs/2504.10063. 
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).
    Computes topological divergences between prompt and response attention 
    graphs to identify hallucination-indicative heads.
    """
    def __init__(
            self, 
            selected_heads: List[Tuple[int, int]] = None,
            num_layers: int = None, 
            num_heads: int = None,
            zero_output: Literal["prompt", "response"] = "prompt", 
            n_jobs: int = 16, 
            critical_size: int = 768
    ):
        """
        Initializes TopologicalDivergence estimator.

        Parameters:
            selected_heads (List[Tuple[int, int]]): List of attention heads to calculate MTopDiv for. 
                First integer is layer index, second is head index. 
                If not provided or empty, all heads will be used.
            zero_out Literal["prompt", "response"]: Determines whether to zero out distances between 
                prompt tokens or response tokens. 
                If not provided or empty, prompt tokens will be zeroed out.
            n_jobs (int): Number of jobs for parallel processing. Default: 16.
            critical_size (int): Size threshold for parallel processing. 
                If the sequence length exceeds this size, n_jobs will be limited to 8. Default: 768.
        """
        super().__init__(["greedy_tokens", "forwardpass_attention_weights"], "sequence")
        
        if selected_heads is None or len(selected_heads) == 0:
            # If no specific heads are selected, use all heads
            if num_layers is None or num_heads is None:
                raise ValueError(
                    "If no specific heads are specified in 'selected_heads', "
                    "'num_layers' and 'num_heads' must be provided."
                )
            selected_heads = [
                (layer, head)
                for layer in range(num_layers)
                for head in range(num_heads)
            ]
        selected_heads_dict = defaultdict(list)
        for layer, head in selected_heads:
            selected_heads_dict[layer].append(head)
        self.selected_heads = selected_heads_dict
        self.zero_out = zero_output
        self.n_jobs = n_jobs
        self.critical_size = critical_size
        
    def __str__(self):
        return "TopologicalDivergence"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculates sequence-wise MTopDiv scores for selected heads of attention masks. 
        Returns the mean MTopDiv score for each input text. 

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, consisting of:
                * tokenized model generations for each input text in 'greedy_tokens'
                * attention scores of shape [batch_size, num_layers, num_heads, seq_len, seq_len]
                    in 'forwardpass_attention_weights'
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        responses = stats["greedy_tokens"]
        attention_weights_batch = stats["forwardpass_attention_weights"]

        layers = sorted(self.selected_heads.keys())
        mtopdivs_batch = []
        for response, attention_weights in zip(responses, attention_weights_batch):
            padding_length = np.isnan(attention_weights[0, 0, 0]).sum()
            response_length = len(response)
            for layer in layers:
                mtopdivs_batch.append([])
                heads = self.selected_heads[layer]
                selected_attention_weights = torch.from_numpy(
                    attention_weights[layer, heads, :-padding_length, :-padding_length]
                ).float()
                distance_matrices = transform_attention_scores_to_distances(
                    selected_attention_weights, self.zero_out, response_length
                )
                if IS_PARALLEL_AVAILABLE:
                    if selected_attention_weights.shape[-1] <= self.critical_size:
                        n_jobs = min(self.n_jobs, 8) if n_jobs > 0 else 8
                    else:
                        n_jobs = self.n_jobs
                    mtopdivs = list(
                        *Parallel(n_jobs=n_jobs)(
                            delayed(transform_distances_to_mtopdiv)(distance_matrice)
                            for distance_matrice in distance_matrices
                        )
                    )
                else:
                    mtopdivs = list(map(
                        transform_distances_to_mtopdiv, distance_matrices
                    ))
                mtopdivs_batch[-1].extend(mtopdivs)
        mtopdivs_batch = np.array(mtopdivs_batch, dtype=np.float)  

        return np.mean(mtopdivs_batch, axis=1)
