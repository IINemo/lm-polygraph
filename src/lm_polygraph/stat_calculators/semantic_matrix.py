import numpy as np

import itertools
from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.estimators.common import DEBERTA
import torch.nn as nn
import torch

softmax = nn.Softmax(dim=1)

class SemanticMatrixCalculator(StatCalculator):
    def __init__(self):
        super().__init__(['semantic_matrix_entail',
                          'semantic_matrix_contra',
                          'semantic_matrix_classes'],
                         ['blackbox_sample_texts'])
        DEBERTA.setup()

    def __call__(self, dependencies: Dict[str, np.array],
                       texts: List[str],
                       model: WhiteboxModel,
                       max_new_tokens: int = 100) -> Dict[str, np.ndarray]:
        deberta_batch_size = dependencies['deberta_batch_size']
        batch_texts = dependencies['blackbox_sample_texts']

        batch_pairs = []
        batch_invs = []
        batch_counts = []
        for texts in batch_texts:
            # Sampling from LLM often produces significant number of identical
            # outputs. We only need to score pairs of unqiue outputs
            unique_texts, inv = np.unique(texts, return_inverse=True)
            batch_pairs.append(itertools.product(unique_texts, unique_texts))
            batch_invs.append(inv)
            batch_counts.append(len(unique_texts))

        device = DEBERTA.device 
        ent_id = DEBERTA.deberta.config.label2id['ENTAILMENT']
        contra_id = DEBERTA.deberta.config.label2id['CONTRADICTION']

        softmax = nn.Softmax(dim=1)
        
        E = []
        C = []
        P = []

        for i, pairs in enumerate(batch_pairs):
            encoded = DEBERTA.deberta_tokenizer.batch_encode_plus(pairs, padding=True, return_tensors='pt').to(device)
            dl = torch.utils.data.DataLoader(dataset, batch_size=deberta_batch_size)
            probs = []
            for batch in dl:
                logits = DEBERTA.deberta(**batch).logits.detach().to(device)
                probs.append(softmax(logits).cpu().detach())
            probs = torch.stack(probs, dim=0]

            entail_probs = probs[:, ent_id]
            contra_probs = probs[:, contra_id]
            class_preds = probs.argmax(-1)

            unique_mat_shape = (batch_counts[i], batch_counts[i])

            unique_E = entail_probs.view(unique_mat_shape).numpy()
            unique_C = contra_probs.view(unique_mat_shape).numpy()
            unique_P = class_preds.view(unique_mat_shape).numpy()
            
            inv = batch_invs[i]
            
            # Recover full matrices from unques by gathering along both axes
            # using inverse index
            E.append(unique_E[inv, :][:, inv])
            C.append(unique_C[inv, :][:, inv])
            P.append(unique_P[inv, :][:, inv])

        E = np.stack(E)
        C = np.stack(C)
        P = np.stack(P)

        return {'semantic_matrix_entail': E,
                'semantic_matrix_contra': C,
                'semantic_matrix_classes': P}
