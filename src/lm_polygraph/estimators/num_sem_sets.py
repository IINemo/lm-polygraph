import numpy as np

from typing import Dict, Literal

from .estimator import Estimator
from .common import DEBERTA, compute_sim_score
import torch.nn as nn


softmax = nn.Softmax(dim=1)

def _get_pairs(lst):
        pairs = []
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                pairs.append((lst[i], lst[j], i, j))
        return pairs

def find_connected_components(graph):
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


class NumSemSets(Estimator):
    def __init__(
            self,
            batch_size: int = 10,
            verbose: bool = False
    ):
        """
        A number of semantic sets in response (higher = bigger uncertainty).

        """
        super().__init__(['blackbox_sample_texts'], 'sequence')
        self.batch_size = batch_size
        DEBERTA.setup()
        self.verbose = verbose
        self.device = DEBERTA.device 

    def __str__(self):
        return f'NumSemSets'

    def get_pairs_semsets(self, lst):
        pairs = []
        for i in range(len(lst) - 1):
            pairs.append((lst[i], lst[i + 1]))
        return pairs
    
    def U_NumSemSets(self, answers):

        num_sets = len(answers)

        W_entail_forward = np.eye(len(answers))
        W_entail_backward = np.eye(len(answers))
        W_contra_forward = np.eye(len(answers))
        W_contra_backward = np.eye(len(answers))

        pairs = _get_pairs(answers)
        device = DEBERTA.device 

        softmax = nn.Softmax(dim=1)

        forward_texts = [pair[:2] for pair in pairs]
        backward_texts = [pair[:2][::-1] for pair in pairs]

        encoded_forward_batch = DEBERTA.deberta_tokenizer.batch_encode_plus(forward_texts, padding=True, return_tensors='pt').to(device)
        encoded_backward_batch = DEBERTA.deberta_tokenizer.batch_encode_plus(backward_texts, padding=True, return_tensors='pt').to(device)

        logits_forward_batch = DEBERTA.deberta(**encoded_forward_batch).logits.detach().to(device)
        logits_backward_batch = DEBERTA.deberta(**encoded_backward_batch).logits.detach().to(device)

        probs_forward_batch = softmax(logits_forward_batch)
        probs_backward_batch = softmax(logits_backward_batch)

        for n, (sentence_1, sentence_2, i, j) in enumerate(pairs):
            probs_forward = probs_forward_batch[n].unsqueeze(0)
            probs_backward = probs_backward_batch[n].unsqueeze(0)

            a_nli_entail_forward = probs_forward[0][2]
            a_nli_entail_backward = probs_backward[0][2]

            a_nli_contra_forward = probs_forward[0][0]
            a_nli_contra_backward = probs_backward[0][0]
            
            forward_entail_w = a_nli_entail_forward.item()
            backward_entail_w = a_nli_entail_backward.item()

            forward_contra_w = a_nli_contra_forward.item()
            backward_contra_w = a_nli_contra_backward.item()

            W_entail_forward[i, j] = forward_entail_w
            W_entail_backward[i, j] = backward_entail_w
            W_contra_forward[i, j] = forward_contra_w
            W_contra_backward[i, j] = backward_contra_w

        W = (W_entail_forward > W_contra_forward).astype(int) * (W_entail_backward > W_contra_backward).astype(int)

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
        connected_components = find_connected_components(graph)

        # Calculate the number of connected components
        num_components = len(connected_components)
        
        return num_components

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        res = []
        for answers in stats['blackbox_sample_texts']:
            if self.verbose:
                print(f"generated answers: {answers}")
            res.append(self.U_NumSemSets(answers))

        return np.array(res)
