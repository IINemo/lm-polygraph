import numpy as np

from typing import Dict, List, Tuple, Callable
from nltk.corpus import stopwords

from lm_polygraph.estimators.estimator import Estimator
from lm_polygraph.stat_calculators.extract_claims import Claim


def nltk_stopword(t: str):
    """Checks if a word is functional using NLTK's English stopwords list."""
    return t in stopwords.words("english")


class ClaimConditionedProbabilityClaim(Estimator):
    """
    Estimator for calculating the claim-conditioned probability (CCP) of claims,
    as proposed in the paper https://arxiv.org/abs/2403.04696.

    CCP measures the probability of the claim meaning, conditioned on its type.
    It leverages NLI to assess the relationship between generated tokens and
    their alternatives.

    For each non-functional token in the generated text, CCP calculates:

    P(entail) / (P(entail) + P(contra))

    where:
     - P(entail) is the sum of probabilities of tokens at the current position that
       are NLI-entailed by the greedy-generated token.
     - P(contra) is the sum of probabilities of tokens NLI-contradicting the
       greedy-generated token.

    Two NLI context modes are supported:

     - "no_context": Utilizes NLI probabilities between two tokens in their
        textual representation without considering the claim.
     - "fact_pref": Utilizes NLI probabilities between two tokens, but both are
        prefixed with the claim prefix to provide context for the NLI model.

    For entailment-contradiction classification, refer to the NLI model used in
    `lm_polygraph.stat_calculators.greedy_alternatives_nli.GreedyAlternativesNLICalculator`.
    """

    def __init__(
        self,
        nli_context: str = "no_context",
        is_stopword: Callable[[str], bool] = nltk_stopword,
    ):
        """
        Initializes the ClaimConditionedProbabilityClaim estimator.

        Args:
            nli_context (str): The NLI context mode to use ("no_context" or "fact_pref").
            is_stopword (Callable[[str], bool]): Function to determine if a token is a
                functional word (skipped when aggregating token-level uncertainties).
        """
        assert nli_context in ["no_context", "fact_pref"]
        self.nli_context = nli_context
        self.is_stopword = is_stopword
        dependencies = [
            "claims",
            "greedy_tokens_alternatives",
        ]
        if nli_context == "no_context":
            dependencies.append("greedy_tokens_alternatives_nli")
        else:
            dependencies.append("greedy_tokens_alternatives_fact_pref_nli")
        super().__init__(dependencies, "claim")

    def __str__(self):
        return "CCP_claim_{}".format(self.nli_context)

    def _reduce(self, logprobs: list[float]):
        """Reduces a list of token-level CCP log probabilities to a single probability."""
        return np.exp(np.sum(logprobs))

    def _combine_nli(self, forward: str, backward: str):
        """
        Combines two NLI predictions NLI(x, y) and NLI(y, x) into a single prediction.

        Prioritizes "entail" or "contra" if present, otherwise returns "neutral".
        """
        if forward == backward:
            return forward
        if all(x in [forward, backward] for x in ["entail", "contra"]):
            return "neutral"
        for x in ["entail", "contra"]:
            if x in [forward, backward]:
                return x
        return "neutral"

    def _token_ccp(
        self,
        token_alternatives: List[Tuple[str, float]],
        token_alternatives_nli: List[List[str]],
    ):
        """
        Calculates the logarithm of the token-level CCP for a single token.

        Args:
            token_alternatives: List of top tokens and their log probabilities at
               the current position.
            token_alternatives_nli: NLI matrix between all pairs of tokens from
                token_alternatives.

        Returns:
            The log CCP for the token, calculated as log(P(entail) / (P(entail) + P(contra)))
        """
        entail_logprobs, entail_words = [], []
        contra_logprobs, contra_words = [], []
        for i in range(len(token_alternatives)):
            word_alt, logprob = token_alternatives[i]
            if self.is_stopword(token_alternatives[0][0]) or i == 0:
                nli = "entail"
            else:
                nli = self._combine_nli(
                    token_alternatives_nli[0][i],
                    token_alternatives_nli[i][0],
                )
            if nli == "entail":
                entail_logprobs.append(logprob)
                entail_words.append(word_alt)
            elif nli == "contra":
                contra_logprobs.append(logprob)
                contra_words.append(word_alt)
        entail_logprob = np.logaddexp.reduce(entail_logprobs)
        total_logprob = np.logaddexp.reduce(entail_logprobs + contra_logprobs)
        return entail_logprob - total_logprob

    def _claim_ccp_no_context(
        self,
        alternatives: List[List[List[Tuple[str, float]]]],
        alternatives_nli: List[List[List[List[str]]]],
        claims: List[List[Claim]],
    ) -> List[List[float]]:
        """
        Calculates the CCP for all claims using the "no_context" NLI option.

        This method iterates through each sample, calculating the CCP for each
        claim associated with that sample. It leverages the NLI relationships
        between generated tokens and their alternatives without considering
        the claim itself in the NLI computation.

        Args:
            alternatives: List of top tokens and their log probabilities, structured as:
                [n_samples, n_tokens_in_sample, n_alternatives]
            alternatives_nli: NLI matrix between token alternatives, structured as:
                [n_samples, n_tokens_in_sample, n_alternatives, n_alternatives]
            claims: List of all claims, structured as:
                [n_samples, n_claims_in_sample]

        Returns:
            List of negative log CCP values for each claim, structured as:
                [n_samples, n_claims_in_sample]
        """
        claim_ue = []
        for s_alternatives, s_alternatives_nli, s_claims in zip(
            alternatives,
            alternatives_nli,
            claims,
        ):
            sample_ccp = []
            claim_ue.append([])
            for token_alternatives, token_alternatives_nli in zip(
                s_alternatives,
                s_alternatives_nli,
            ):
                sample_ccp.append(
                    self._token_ccp(
                        token_alternatives,
                        token_alternatives_nli,
                    ),
                )
            sample_ccp = np.array(sample_ccp)
            for claim in s_claims:
                tokens = np.array(claim.aligned_token_ids)
                claim_ue[-1].append(-self._reduce(sample_ccp[tokens]))
        return claim_ue

    def _claim_ccp_fact_pref(
        self,
        alternatives: List[List[List[Tuple[str, float]]]],
        alternatives_nli: List[List[List[List[str]]]],
        claims: List[List[Claim]],
    ) -> List[List[float]]:
        """
        Calculates the CCP for all claims using the "fact_pref" NLI option.

        This method incorporates the claim as a prefix when computing NLI
        relationships between the generated token and its alternatives.
        It iterates through each claim within each sample to calculate the
        claim-specific CCP.

        Args:
            alternatives: List of top tokens and their log probabilities, structured as:
                [n_samples, n_tokens_in_sample, n_alternatives]
            alternatives_nli: NLI matrix between token alternatives (with claim prefix),
                structured as: [n_samples, n_claims_in_sample, n_claim_tokens, n_alternatives, n_alternatives]
            claims: List of all claims, structured as:
                [n_samples, n_claims_in_sample]

        Returns:
            List of negative log CCP values for each claim, structured as:
                [n_samples, n_claims_in_sample]
        """
        claim_ue = []
        for s_alternatives, s_alternatives_nli, s_claims in zip(
            alternatives,
            alternatives_nli,
            claims,
        ):
            claim_ue.append([])
            for claim, claim_nlis in zip(
                s_claims,
                s_alternatives_nli,
            ):
                assert len(claim_nlis) == len(claim.aligned_token_ids)
                token_alternatives = []
                for t in claim.aligned_token_ids:
                    token_alternatives.append(s_alternatives[t])
                token_alternatives_nli = claim_nlis
                token_ccps = []
                for token_alt, token_nli in zip(
                    token_alternatives,
                    token_alternatives_nli,
                ):
                    token_ccps.append(self._token_ccp(token_alt, token_nli))
                claim_ue[-1].append(-self._reduce(token_ccps))
        return claim_ue

    def __call__(self, stats: Dict[str, object]) -> List[List[float]]:
        """
        Calculates and returns the CCP for all claims in the provided statistics.

        This method dispatches the appropriate CCP calculation method
        ("no_context" or "fact_pref") based on the configured `nli_context`.

        Args:
            stats: A dictionary containing the required statistics, including:
                - "greedy_tokens_alternatives": Token alternatives and their probabilities
                - "greedy_tokens_alternatives_nli" or "greedy_tokens_alternatives_fact_pref_nli": NLI matrices
                - "claims": The list of claims

        Returns:
            A list of lists containing the negative log CCP for each claim, structured as:
                [num_samples, num_claims_in_sample]
        """
        alternatives = stats["greedy_tokens_alternatives"]
        claims = stats["claims"]
        if self.nli_context == "no_context":
            alternatives_nli = stats["greedy_tokens_alternatives_nli"]
            return self._claim_ccp_no_context(
                alternatives,
                alternatives_nli,
                claims,
            )
        elif self.nli_context == "fact_pref":
            alternatives_nli = stats["greedy_tokens_alternatives_fact_pref_nli"]
            return self._claim_ccp_fact_pref(
                alternatives,
                alternatives_nli,
                claims,
            )
        else:
            raise Exception(f"Unsupported argument nli_context={self.nli_context}")
