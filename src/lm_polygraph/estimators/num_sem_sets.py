import numpy as np

from typing import Dict

from .estimator import Estimator
from .common import DEBERTA
import torch.nn as nn

softmax = nn.Softmax(dim=1)


class NumSemSets(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "Number of Semantic Sets" as provided in the paper https://arxiv.org/abs/2305.19187.
    Works with both whitebox and blackbox models (initialized using
    lm_polygraph.utils.model.BlackboxModel/WhiteboxModel).
    """

    def __init__(
        self,
        batch_size: int = 10,
        verbose: bool = False,
    ):
        """
        Parameters:
            batch_size (int): batch size for Deberta model.
        """
        super().__init__(
            [
                "semantic_matrix_entail",
                "semantic_matrix_contra",
                "blackbox_sample_texts",
            ],
            "sequence",
        )
        self.batch_size = batch_size
        DEBERTA.setup()
        self.verbose = verbose

    def __str__(self):
        return "NumSemSets"

    def find_connected_components(self, graph):
        def dfs(node, component):
            visited[node] = True
            component.append(node)

            for neighbor in graph[node]:
                if not visited[neighbor]:
                    dfs(neighbor, component)

        visited = [False] * len(graph)
        components = []

        for i in range(len(graph)):
            if not visited[i]:
                component = []
                dfs(i, component)
                components.append(component)

        return components

    def U_NumSemSets(self, i, stats):
        # We have forward upper triangular and backward in lower triangular
        # parts of the semantic matrices
        W_entail = stats["semantic_matrix_entail"][i, :, :]
        W_contra = stats["semantic_matrix_contra"][i, :, :]

        # We check that for every pairing (both forward and backward)
        # the condition satisfies
        W = (W_entail > W_contra).astype(int)
        # Multiply by it's transpose to get symmetric matrix of full condition
        W = W * np.transpose(W)
        # Take upper triangular part with no diag
        W = np.triu(W, k=1)

        a = [[i] for i in range(W.shape[0])]

        # Iterate through each row in 'W' and update the corresponding row in 'a'
        for i, row in enumerate(W):
            # Find the indices of non-zero elements in the current row
            non_zero_indices = np.where(row != 0)[0]

            # Append the non-zero indices to the corresponding row in 'a'
            a[i].extend(non_zero_indices.tolist())

        # Create an adjacency list representation of the graph
        graph = [[] for _ in range(len(a))]
        for sublist in a:
            for i in range(len(sublist) - 1):
                graph[sublist[i]].append(sublist[i + 1])
                graph[sublist[i + 1]].append(sublist[i])

        # Find the connected components
        connected_components = self.find_connected_components(graph)

        # Calculate the number of connected components
        num_components = len(connected_components)

        return num_components

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the number of semantic sets for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * generated samples in 'blackbox_sample_texts',
                * matrix with corresponding semantic similarities in
                  'semantic_matrix_entail' and 'semantic_matrix_contra'
        Returns:
            np.ndarray: number of semantic sets for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        res = []
        for i, answers in enumerate(stats["blackbox_sample_texts"]):
            if self.verbose:
                print(f"generated answers: {answers}")
            res.append(self.U_NumSemSets(i, stats))

        return np.array(res)
