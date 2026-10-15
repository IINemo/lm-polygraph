"""Tests for the Spilled Energy method (EnergyCalculator + SpilledEnergy estimator).

Both tests run on CPU in seconds. The only network access is a ~2.4M parameter
random-weight Qwen2 stub; that test is skipped automatically when the model
cannot be fetched (offline CI), while the synthetic test needs no network at all.
"""

import numpy as np
import pytest
import torch

from lm_polygraph.estimators import SpilledEnergy
from lm_polygraph.stat_calculators.energy import EnergyCalculator

TINY_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


# ---------------------------------------------------------------------------
# Test 1: the teacher-forced raw-logit pass is consistent with the generation
# ---------------------------------------------------------------------------
# EnergyCalculator re-runs the model teacher-forced to recover raw logits.
# If its indexing is right, then for every generated token j:
#
#     log_softmax(theta_j)[id(x_j)] == theta_j[id(x_j)] - Z_j
#                                   == energy_token_logits[j] - energy_lse[j]
#
# must equal greedy_log_likelihoods[j] produced during generation. This pins the
# off-by-one in the teacher-forcing alignment, which is the single easiest thing
# to get wrong here.


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lm_polygraph.utils.model import WhiteboxModel
    from lm_polygraph.utils.generation_parameters import GenerationParameters

    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            TINY_MODEL, attn_implementation="eager"
        )
        tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL, padding_side="left")
    except Exception as e:  # offline / hub unavailable
        pytest.skip(f"tiny stub model unavailable: {e}")

    hf_model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return WhiteboxModel(
        hf_model,
        tokenizer,
        model_path=TINY_MODEL,
        model_type="CausalLM",
        generation_parameters=GenerationParameters(do_sample=False, num_beams=1),
    )


def test_teacher_forced_logits_match_greedy_log_likelihoods(tiny_model):
    """log_softmax of the teacher-forced pass == greedy_log_likelihoods (1e-3)."""
    from lm_polygraph.stat_calculators.greedy_probs import GreedyProbsCalculator

    texts = [
        "Q: What is the capital of France?\nA:",
        "Q: Who wrote Hamlet?\nA:",
    ]

    greedy = GreedyProbsCalculator(output_attentions=False, output_hidden_states=False)
    gstats = greedy({}, texts, tiny_model, max_new_tokens=6)

    energy = EnergyCalculator()
    estats = energy(
        {"greedy_tokens": gstats["greedy_tokens"]}, texts, tiny_model, max_new_tokens=6
    )

    tok_logits = estats["energy_token_logits"]
    lse = estats["energy_lse"]
    ref_ll = gstats["greedy_log_likelihoods"]

    for i in range(len(texts)):
        n = len(gstats["greedy_tokens"][i])
        assert len(tok_logits[i]) == n
        assert len(lse[i]) == n + 1, "energy_lse must carry one extra step"

        # theta[id] - Z  ==  log_softmax(theta)[id]
        recovered = np.asarray(tok_logits[i], dtype=np.float64) - np.asarray(
            lse[i][:n], dtype=np.float64
        )
        expected = np.asarray(ref_ll[i], dtype=np.float64)

        np.testing.assert_allclose(recovered, expected, atol=1e-3, rtol=0)


def test_energy_lse_is_a_true_logsumexp(tiny_model):
    """energy_lse must be the raw log-partition, i.e. strictly non-zero.

    Guards against silently reading the log_softmax'd scores, where logsumexp
    over the vocabulary is identically 0 and all energy information is lost.
    """
    from lm_polygraph.stat_calculators.greedy_probs import GreedyProbsCalculator

    texts = ["Q: What is the capital of France?\nA:"]
    greedy = GreedyProbsCalculator(output_attentions=False, output_hidden_states=False)
    gstats = greedy({}, texts, tiny_model, max_new_tokens=5)

    estats = EnergyCalculator()(
        {"greedy_tokens": gstats["greedy_tokens"]}, texts, tiny_model, max_new_tokens=5
    )
    lse = np.asarray(estats["energy_lse"][0], dtype=np.float64)

    assert np.all(np.isfinite(lse))
    assert np.abs(lse).min() > 1e-6, (
        "log-partition is ~0 -- EnergyCalculator is reading normalised log-probs, "
        "not raw logits"
    )


# ---------------------------------------------------------------------------
# Test 2: hand-computed spilled energy on a synthetic logit tensor
# ---------------------------------------------------------------------------


def test_spilled_energy_matches_hand_computation():
    """dE_j = Z_{j+1} - theta_j[id(x_j)], verified against arithmetic done by hand."""
    # 3 decoding steps over a 5-token vocabulary; the generation is 2 tokens long,
    # so we need N + 1 = 3 rows of logits.
    theta = np.array(
        [
            [2.00, 1.00, 0.50, 0.00, -1.00],
            [0.00, 3.00, 1.00, 0.50, 0.25],
            [1.00, 1.00, 1.00, 1.00, 1.00],
        ],
        dtype=np.float64,
    )
    generated = [0, 1]  # sampled token ids at steps 0 and 1

    # Hand-computed log-partitions.
    # Row 2 is uniform: logsumexp = log(5 * e^1) = 1 + log(5).
    lse = np.array([float(torch.logsumexp(torch.tensor(r), dim=-1)) for r in theta])
    assert lse[2] == pytest.approx(1.0 + np.log(5.0), abs=1e-9)

    tok_logits = np.array([theta[0, generated[0]], theta[1, generated[1]]])
    assert tok_logits.tolist() == [2.0, 3.0]

    # dE_0 = Z_1 - theta_0[0]; dE_1 = Z_2 - theta_1[1]
    expected_delta = np.array([lse[1] - 2.0, lse[2] - 3.0])

    stats = {"energy_token_logits": [tok_logits], "energy_lse": [lse]}

    got_max = SpilledEnergy(variant="spilled", pooling="max")(stats)
    got_min = SpilledEnergy(variant="spilled", pooling="min")(stats)
    got_mean = SpilledEnergy(variant="spilled", pooling="mean")(stats)

    assert got_max[0] == pytest.approx(expected_delta.max(), abs=1e-9)
    assert got_min[0] == pytest.approx(expected_delta.min(), abs=1e-9)
    assert got_mean[0] == pytest.approx(expected_delta.mean(), abs=1e-9)

    # marginal energy E^m_j = -Z_j, taken at each generated token's own step
    expected_marginal = -lse[:2]
    got_marginal = SpilledEnergy(variant="marginal", pooling="min")(stats)
    assert got_marginal[0] == pytest.approx(expected_marginal.min(), abs=1e-9)

    # scaled spilled energy: |E^m| * dE
    expected_scaled = np.abs(-lse[:2]) * expected_delta
    got_scaled = SpilledEnergy(variant="scaled_spilled", pooling="max")(stats)
    assert got_scaled[0] == pytest.approx(expected_scaled.max(), abs=1e-9)

    # sign flip must negate the score exactly
    got_neg = SpilledEnergy(variant="spilled", pooling="max", sign=-1)(stats)
    assert got_neg[0] == pytest.approx(-expected_delta.max(), abs=1e-9)


def test_logit_energy_and_the_decomposition_identity():
    """E^l is the raw sampled-token logit, and E^l/E^m recompose the log-likelihood.

    The point of exposing ``logit`` is that the ablation ladder differs by exactly
    one ingredient per rung. That only holds if
        log p(x_j) = theta_j[id] - Z_j = -E^l_j - (-E^m_j) = E^m_j - E^l_j
    which is asserted here on values chosen so the arithmetic is checkable by hand.
    """
    theta = np.array(
        [
            [2.00, 1.00, 0.50, 0.00, -1.00],
            [0.00, 3.00, 1.00, 0.50, 0.25],
            [1.00, 1.00, 1.00, 1.00, 1.00],
        ],
        dtype=np.float64,
    )
    generated = [0, 1]
    lse = np.array([float(torch.logsumexp(torch.tensor(r), dim=-1)) for r in theta])
    tok_logits = np.array([theta[0, generated[0]], theta[1, generated[1]]])
    stats = {"energy_token_logits": [tok_logits], "energy_lse": [lse]}

    # E^l = -theta[id]; pooled with max -> max(-[2, 3]) = -2
    got_logit = SpilledEnergy(variant="logit", pooling="max")(stats)
    assert got_logit[0] == pytest.approx(-2.0, abs=1e-9)
    got_logit_min = SpilledEnergy(variant="logit", pooling="min")(stats)
    assert got_logit_min[0] == pytest.approx(-3.0, abs=1e-9)

    # the identity, per token
    e_l = -tok_logits
    e_m = -lse[:2]
    log_probs = e_m - e_l
    expected_log_probs = tok_logits - lse[:2]
    np.testing.assert_allclose(log_probs, expected_log_probs, atol=1e-12)
    # and those are genuine log-probabilities
    assert np.all(log_probs < 0)


def test_energy_calculator_rejects_non_finite_logits():
    """Non-finite logits must abort loudly, not flow into the energies.

    Regression test for the fp16 overflow seen on the T4: raw logits went inf,
    argmax collapsed to token id 0 ('!' in Qwen's vocab), and the corrupted
    energies would have propagated silently into PRR. Because SpilledEnergy reads
    RAW logits, nothing downstream normalises a NaN away.
    """
    n = 3
    tok_logits = torch.tensor([1.0, 2.0, 3.0])
    lse = torch.tensor([2.0, 3.0, 4.0, 5.0])
    rows = torch.zeros((n + 1, 8))

    # clean input passes
    EnergyCalculator._assert_finite(tok_logits, lse, rows, 0)

    for label, bad_tok, bad_lse in [
        ("inf in sampled-token logit", torch.tensor([1.0, float("inf"), 3.0]), lse),
        ("nan in sampled-token logit", torch.tensor([1.0, float("nan"), 3.0]), lse),
        (
            "inf in log-partition",
            tok_logits,
            torch.tensor([2.0, float("inf"), 4.0, 5.0]),
        ),
    ]:
        with pytest.raises(RuntimeError, match="non-finite"):
            EnergyCalculator._assert_finite(bad_tok, bad_lse, rows, 7)

    # the message must name the sample and point at the cause
    try:
        EnergyCalculator._assert_finite(
            torch.tensor([float("inf"), 2.0, 3.0]), lse, rows, 42
        )
    except RuntimeError as e:
        msg = str(e)
        assert "sample 42" in msg
        assert "fp16" in msg


def test_spilled_energy_naming_and_validation():
    """__str__ must disambiguate configurations; bad args must raise."""
    assert str(SpilledEnergy("logit", "max")) == "SpilledEnergy_logit_max"
    assert str(SpilledEnergy("spilled", "max")) == "SpilledEnergy_spilled_max"
    assert str(SpilledEnergy("marginal", "min")) == "SpilledEnergy_marginal_min"
    assert (
        str(SpilledEnergy("spilled", "max", sign=-1)) == "SpilledEnergy_spilled_max_neg"
    )
    assert str(SpilledEnergy("spilled", "max")) != str(SpilledEnergy("spilled", "min"))

    for bad in (
        dict(variant="nonsense"),
        dict(pooling="nonsense"),
        dict(sign=0),
    ):
        with pytest.raises(ValueError):
            SpilledEnergy(**bad)


def test_spilled_energy_handles_degenerate_samples():
    """Empty generations yield NaN rather than crashing the whole run."""
    stats = {
        "energy_token_logits": [np.array([]), np.array([1.0])],
        "energy_lse": [np.array([]), np.array([2.0, 3.0])],
    }
    out = SpilledEnergy(variant="spilled", pooling="max")(stats)
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(3.0 - 1.0, abs=1e-9)


def test_terminator_exclusion_trims_the_pooling_window():
    """exclude_terminator must drop exactly the trailing terminator tokens.

    On a TriviaQA generation the window is ~4 tokens and the trailing newline
    plus the EOS are two of them, so half the pooled values are maximally
    predictable tokens whose scores are near-constant across samples and
    unrelated to correctness.
    """
    # 4 generated tokens; the last two are the terminator pair
    tok_logits = np.array([5.0, 6.0, 20.0, 25.0])  # terminators have high logits
    lse = np.array([7.0, 8.0, 21.0, 26.0, 27.0])
    stats_in = {
        "energy_token_logits": [tok_logits],
        "energy_lse": [lse],
        "energy_trailing_terminators": [2],
    }

    # E^l = -tok_logits. Including terminators, max is -5.0; excluding, still -5.0,
    # but MIN changes from -25.0 (a terminator) to -6.0 (a real answer token).
    incl_min = SpilledEnergy(variant="logit", pooling="min")(stats_in)[0]
    excl_min = SpilledEnergy(variant="logit", pooling="min", exclude_terminator=True)(
        stats_in
    )[0]
    assert incl_min == pytest.approx(-25.0)
    assert excl_min == pytest.approx(-6.0), "min pooling still sees the terminator"

    # mean is diluted by the terminators too
    incl_mean = SpilledEnergy(variant="logit", pooling="mean")(stats_in)[0]
    excl_mean = SpilledEnergy(variant="logit", pooling="mean", exclude_terminator=True)(
        stats_in
    )[0]
    assert incl_mean == pytest.approx(-14.0)
    assert excl_mean == pytest.approx(-5.5)

    # naming must distinguish the two, or UEManager would reject the pair
    assert str(SpilledEnergy(variant="logit", pooling="min")) != str(
        SpilledEnergy(variant="logit", pooling="min", exclude_terminator=True)
    )

    # a window that is ALL terminators must still yield a score, not an empty pool
    degenerate = {
        "energy_token_logits": [np.array([3.0, 4.0])],
        "energy_lse": [np.array([5.0, 6.0, 7.0])],
        "energy_trailing_terminators": [2],
    }
    out = SpilledEnergy(variant="logit", pooling="min", exclude_terminator=True)(
        degenerate
    )[0]
    assert np.isfinite(out), "over-trimmed to an empty window"


def test_trailing_terminator_count_is_correct():
    """The counter must stop at the first real token and never empty the window."""
    from lm_polygraph.stat_calculators.energy import EnergyCalculator

    class FakeTok:
        eos_token_id, pad_token_id = 99, 98

        def decode(self, ids):
            return {1: "fr", 2: "iday", 3: "\n", 4: " ", 5: "x"}.get(ids[0], "?")

    tok = FakeTok()
    count = EnergyCalculator._count_trailing_terminators
    assert count([1, 2, 3, 99], tok) == 2  # newline + eos
    assert count([1, 2, 99], tok) == 1  # eos only
    assert count([1, 2], tok) == 0  # no terminator
    assert count([1, 3, 4, 99], tok) == 3  # newline, space, eos
    assert count([3, 3, 3], tok) == 2  # all terminators -> keeps one
    assert count([1, 99, 2], tok) == 0  # eos mid-sequence is not trailing
