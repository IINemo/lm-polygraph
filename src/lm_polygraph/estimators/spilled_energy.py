import numpy as np

from typing import Dict

from .estimator import Estimator

VARIANTS = ("logit", "marginal", "spilled", "scaled_spilled")
POOLINGS = ("min", "max", "mean")


class SpilledEnergy(Estimator):
    """
    Spilled Energy uncertainty estimator.

    Reference:
        "Spilled Energy in Large Language Models", Minut, Dewidar & Masi,
        ICLR 2026. Reference code: github.com/OmnAI-Lab/spilled-energy

    Core idea
    ---------
    The final softmax classifier is read as an Energy-Based Model. Writing the
    conditional ``p(x_i | x_{i-1:1})`` as a ratio of two EBMs yields two energies
    that are computable directly from the raw logits ``theta``:

    * logit energy     ``E^l_j = -theta_j[id(x_j)]``          (negative sampled-token logit)
    * marginal energy  ``E^m_j = -logsumexp_k theta_j[k]``     (negative log-partition)

    By the chain rule these should cancel between *adjacent* decoding steps. They
    do not, and the residual is the spilled energy. Following the authors'
    implementation, for generated token ``j``::

        dE_j = Z_{j+1} - theta_j[id(x_j)]

    where ``Z = logsumexp_k theta[k]``. (The paper's Eq. (8) writes this with the
    opposite sign; the ``sign`` parameter below exists so the orientation is fixed
    empirically rather than by trusting either source -- see ``sign``.)

    Four score variants are supported, all training-free, logits-only and
    computed from a single teacher-forced pass (see ``EnergyCalculator``):

    * ``logit``          -- the logit energy ``E^l``
    * ``marginal``       -- the marginal energy ``E^m``
    * ``spilled``        -- the spilled energy ``dE``
    * ``scaled_spilled`` -- ``|E^m| * dE``

    ``logit`` is the paper's own ``E^l`` baseline -- the negative raw logit of the
    sampled token, i.e. classic logit confidence. It is included because it is one
    of the two energies the method is derived from, so the implementation would be
    incomplete without it, and because lm-polygraph cannot otherwise express it:
    every other statistic is ``log_softmax``-normalised, which subtracts the
    log-partition and destroys the raw logit (see ``EnergyCalculator``). Having it
    here makes the decomposition testable -- ``logit`` vs ``marginal`` vs
    ``spilled`` differ by exactly one ingredient at a time.

    Scores are pooled across the answer span with ``min``, ``max`` or ``mean``.
    """

    def __init__(
        self,
        variant: str = "spilled",
        pooling: str = "max",
        sign: int = 1,
        exclude_terminator: bool = False,
    ):
        """
        Parameters:
            variant (str): one of 'logit', 'marginal', 'spilled',
                'scaled_spilled'; see the class docstring.
            pooling (str): one of 'min', 'max', 'mean'; pooling across the span.
            sign (int): +1 or -1, multiplied into the final score. The Estimator
                contract requires **higher = more uncertain**; which orientation
                satisfies that is an empirical question for each variant, so it is
                an explicit parameter rather than a hard-coded guess.
            exclude_terminator (bool): drop the trailing terminator tokens (EOS,
                and the newline emitted by ``stop_strings``) from the pooling
                window. They are maximally predictable and, on short answers,
                make up a large share of it.
        """
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
        if sign not in (1, -1):
            raise ValueError(f"sign must be +1 or -1, got {sign!r}")

        deps = ["energy_token_logits", "energy_lse"]
        if exclude_terminator:
            deps.append("energy_trailing_terminators")
        super().__init__(deps, "sequence")
        self.variant = variant
        self.pooling = pooling
        self.sign = sign
        self.exclude_terminator = exclude_terminator

    def __str__(self):
        sign_tag = "" if self.sign == 1 else "_neg"
        term_tag = "_noterm" if self.exclude_terminator else ""
        return f"SpilledEnergy_{self.variant}_{self.pooling}{sign_tag}{term_tag}"

    def _per_token_scores(self, tok_logits: np.ndarray, lse: np.ndarray) -> np.ndarray:
        """
        Per-token score for one sample.

        ``tok_logits`` has length N (raw logit of each sampled token) and ``lse``
        has length N+1 (log-partition at each step plus the step after the last),
        as produced by EnergyCalculator.
        """
        tok_logits = np.asarray(tok_logits, dtype=np.float64)
        lse = np.asarray(lse, dtype=np.float64)
        n = len(tok_logits)

        if self.variant == "logit":
            # logit energy: negative raw logit of the sampled token
            return -tok_logits

        if self.variant == "marginal":
            # marginal energy at each generated token's own step
            return -lse[:n]

        # adjacent-step residual: Z_{j+1} - theta_j[id(x_j)]
        delta = lse[1 : n + 1] - tok_logits

        if self.variant == "spilled":
            return delta

        # scaled_spilled: |E^m| * dE, with E^m taken at the token's own step
        return np.abs(-lse[:n]) * delta

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Computes the Spilled Energy score for each sample.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which include:
                * 'energy_token_logits': raw logits of the sampled tokens,
                * 'energy_lse': per-step log-partitions (one longer).
        Returns:
            np.ndarray: uncertainty score per sample; higher = more uncertain.
                A sample with an empty generation, or with fewer log-partitions
                than the variant needs, yields ``np.nan`` -- no score is defined
                for it, and substituting one would silently corrupt the ranking.
                Note that the UE metrics drop NaN *targets* but not NaN
                *estimates*, so such samples have to be filtered out upstream of
                scoring.
        """
        all_tok_logits = stats["energy_token_logits"]
        all_lse = stats["energy_lse"]
        trailing = (
            stats["energy_trailing_terminators"]
            if self.exclude_terminator
            else [0] * len(all_tok_logits)
        )

        pool = {"min": np.min, "max": np.max, "mean": np.mean}[self.pooling]

        out = []
        for tok_logits, lse, n_term in zip(all_tok_logits, all_lse, trailing):
            n = len(tok_logits)
            if n == 0 or len(lse) < n + 1:
                out.append(np.nan)
                continue
            scores = self._per_token_scores(tok_logits, lse)
            # Drop the trailing terminator tokens from the pooling window. On a
            # ~4-token TriviaQA generation the newline and the EOS are half of it,
            # and both are maximally predictable, so they carry near-constant
            # scores unrelated to correctness.
            if n_term:
                keep = max(1, len(scores) - int(n_term))
                scores = scores[:keep]
            scores = scores[np.isfinite(scores)]
            out.append(np.nan if scores.size == 0 else float(pool(scores)))

        return self.sign * np.array(out, dtype=np.float64)
