import numpy as np

from typing import Dict
from nltk.corpus import stopwords

from .estimator import Estimator


class ClaimConditionedProbability(Estimator):
    def __init__(self):
        super().__init__(
            [
                "greedy_tokens",
                "greedy_tokens_alternatives",
                "greedy_tokens_alternatives_nli",
            ],
            "sequence",
        )

    def __str__(self):
        return "CCP"

    def _reduce(self, logprobs: list[float]):
        return np.exp(np.sum(logprobs))

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        words = stats["greedy_tokens"]
        alternatives = stats["greedy_tokens_alternatives"]
        alternatives_nli = stats["greedy_tokens_alternatives_nli"]
        prob_nli = []
        for sample_words, sample_alternatives, sample_alternatives_nli in zip(
            words,
            alternatives,
            alternatives_nli,
        ):
            sample_mnlis = []
            for word, word_alternatives, word_alternatives_nli in zip(
                sample_words,
                sample_alternatives,
                sample_alternatives_nli,
            ):
                entail_logprobs, entail_words = [], []
                contra_logprobs, contra_words = [], []
                for i in range(len(word_alternatives)):
                    word_alt, logprob = word_alternatives[i]
                    if i == 0 or word_alternatives_nli[0][i] == "entail":
                        entail_logprobs.append(logprob)
                        entail_words.append(word_alt)
                    elif word_alternatives_nli[0][i] == "contra":
                        contra_logprobs.append(logprob)
                        contra_words.append(word_alt)
                entail_logprob = np.logaddexp.reduce(entail_logprobs)
                total_logprob = np.logaddexp.reduce(entail_logprobs + contra_logprobs)
                sample_mnlis.append(entail_logprob - total_logprob)
            prob_nli.append(self._reduce(sample_mnlis))
        return -np.array(prob_nli)


def nltk_stopword(t: str):
    return t in stopwords.words("english")


class ClaimConditionedProbabilityClaim(Estimator):
    def __init__(self, nli_context: str = "no_context", is_stopword=nltk_stopword):
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
        return np.exp(np.sum(logprobs))

    def _combine_nli(self, forward: str, backward: str):
        if forward == backward:
            return forward
        if all(x in [forward, backward] for x in ["entail", "contra"]):
            return "neutral"
        for x in ["entail", "contra"]:
            if x in [forward, backward]:
                return x
        return "neutral"

    def _token_ccp(self, token_alternatives, token_alternatives_nli):
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

    def _claim_ccp_no_context(self, alternatives, alternatives_nli, claims):
        all_claim_ue = []
        for s_alternatives, s_alternatives_nli, s_claims in zip(
            alternatives,
            alternatives_nli,
            claims,
        ):
            claim_ue = []
            sample_ccp = []
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
                tokens = np.array(claim.aligned_tokens)
                claim_ue.append(-self._reduce(sample_ccp[tokens]))
            
            all_claim_ue.append(claim_ue)
        
        return all_claim_ue

    def _claim_ccp_fact_pref(self, alternatives, alternatives_nli, claims):
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
                assert len(claim_nlis) == len(claim.aligned_tokens)
                token_alternatives = []
                for t in claim.aligned_tokens:
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

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
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
