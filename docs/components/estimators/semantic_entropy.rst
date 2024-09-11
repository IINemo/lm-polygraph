.. _SemanticEntropy:

Semantic Entropy
================

Semantic Entropy approach is adapted from `Kuhn et al. <https://arxiv.org/abs/2302.09664>`_.

The idea is to sample a set of model's responses and partition them into several clusters based on their semantic similarity. In this implementation, clustering is performed on the basis of bi-directional entailment as determined by the NLI model. Default NLI model is `microsoft/deberta-large-mnli`.

A probability distribution is then estimated over the clusters. Two ways of esimating the probability of a cluster are implemented:

- Only unique responses corresponding to each cluster are considered. This prevents cluster probabilities exceeding 1.0 in case there are several identical high-probability responses in a cluster. This is the default behavior of the estimator.
- All generated responses that are assigned to a cluster are considered. This approach follows the `original implementation <https://github.com/lorenzkuhn/semantic_uncertainty>`_. This type of estimation can be enabled by setting `use_unique_responses=False` when initializing the estimator.
