"""EnergyCalculator must produce the same energies batched as unbatched.

This is the batched-vs-unbatched property already asserted for *generation*,
applied to the statistic the whole method rests on. It is the gap that let an
A-vs-C discrepancy reach the tables: run A (batch_size=4) and run C
(batch_size=1) share generations exactly, yet SpilledEnergy_spilled_max
correlated at only rho=0.294 between them, with a median absolute difference of
2.54 and a maximum of 15.06.

dE is an adjacent-step difference, so it is the quantity most sensitive to any
position misalignment or padding leak -- and it is also the quantity that
underperformed. A negative result sitting exactly where an artefact would bite
hardest cannot be reported until this is settled.

Runs on CPU against the tiny random-weight Qwen2 stub. Parameterised over dtype
so it discriminates: a failure in float32 means a STRUCTURAL bug (padding,
position ids, slicing); a failure only in float16 means NUMERICAL instability of
the statistic itself.
"""

import numpy as np
import pytest
import torch

TINY_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

# Deliberately very different lengths, mirroring the real 165-185 token spread.
PROMPTS = [
    "Question: What is the capital of France?\nAnswer:",
    "Question: Who wrote Hamlet, the tragedy set in Denmark?\nAnswer:",
    "Question: Search for Extra-Terrestrial Intelligence is the collective name "
    "for a number of activities to search for extra-terrestrial life using "
    "scientific methods to search for what, specifically?\nAnswer:",
    "Question: Name the Danny Boyle biopic about a climber trapped by a boulder "
    "for more than five days in a canyon in Utah?\nAnswer:",
]


def _model(dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lm_polygraph.utils.model import WhiteboxModel
    from lm_polygraph.utils.generation_parameters import GenerationParameters

    try:
        hf = AutoModelForCausalLM.from_pretrained(
            TINY_MODEL, attn_implementation="eager", torch_dtype=dtype
        )
        tok = AutoTokenizer.from_pretrained(TINY_MODEL, padding_side="left")
    except Exception as e:
        pytest.skip(f"tiny stub unavailable: {e}")
    hf.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return WhiteboxModel(
        hf,
        tok,
        model_path=TINY_MODEL,
        model_type="CausalLM",
        generation_parameters=GenerationParameters(do_sample=False, num_beams=1),
    )


def _energies(model, texts, greedy_tokens):
    from lm_polygraph.stat_calculators.energy import EnergyCalculator

    return EnergyCalculator()(
        {"greedy_tokens": greedy_tokens}, texts, model, max_new_tokens=8
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_energies_are_batch_invariant(dtype):
    """Per-token energies must match between batch_size=1 and batch_size=4."""
    from lm_polygraph.stat_calculators.greedy_probs import GreedyProbsCalculator

    model = _model(dtype)
    greedy = GreedyProbsCalculator(output_attentions=False, output_hidden_states=False)

    # generate once, so both paths score the SAME token sequences
    gen = greedy({}, PROMPTS, model, max_new_tokens=8)
    tokens = gen["greedy_tokens"]

    batched = _energies(model, PROMPTS, tokens)
    singles = [_energies(model, [PROMPTS[i]], [tokens[i]]) for i in range(len(PROMPTS))]

    tol = 1e-3 if dtype == torch.float32 else 5e-2
    worst = 0.0
    for i in range(len(PROMPTS)):
        for key in ("energy_token_logits", "energy_lse"):
            b = np.asarray(batched[key][i], dtype=np.float64)
            s = np.asarray(singles[i][key][0], dtype=np.float64)
            assert b.shape == s.shape, f"{key} shape differs for sample {i}"
            worst = max(worst, float(np.abs(b - s).max()))

    assert worst <= tol, (
        f"[{dtype}] energies differ between batch_size=4 and batch_size=1: "
        f"max |delta| = {worst:.5f} > {tol}. In float32 this is a STRUCTURAL bug "
        f"(padding / position ids / slicing). In float16 it is numerical "
        f"instability of the statistic."
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_spilled_energy_is_batch_invariant(dtype):
    """dE is an adjacent-step difference and so the most alignment-sensitive score."""
    from lm_polygraph.estimators import SpilledEnergy
    from lm_polygraph.stat_calculators.greedy_probs import GreedyProbsCalculator

    model = _model(dtype)
    greedy = GreedyProbsCalculator(output_attentions=False, output_hidden_states=False)
    gen = greedy({}, PROMPTS, model, max_new_tokens=8)
    tokens = gen["greedy_tokens"]

    batched = _energies(model, PROMPTS, tokens)
    singles = [_energies(model, [PROMPTS[i]], [tokens[i]]) for i in range(len(PROMPTS))]
    merged = {
        k: [singles[i][k][0] for i in range(len(PROMPTS))]
        for k in ("energy_token_logits", "energy_lse", "energy_trailing_terminators")
    }

    tol = 1e-3 if dtype == torch.float32 else 1e-1
    for variant in ("logit", "marginal", "spilled", "scaled_spilled"):
        for pooling in ("min", "max", "mean"):
            est = SpilledEnergy(variant=variant, pooling=pooling)
            a, b = est(batched), est(merged)
            delta = float(np.nanmax(np.abs(np.asarray(a) - np.asarray(b))))
            assert delta <= tol, (
                f"[{dtype}] {variant}/{pooling} differs by {delta:.5f} between "
                f"batch_size=4 and batch_size=1 (tol {tol})"
            )


def test_log_prob_identity_holds_on_the_energy_stats():
    """log p = E^m - E^l is exact, so it must hold on the produced statistics.

    Any residual above fp16 noise means the teacher-forced pass is not scoring the
    same distribution the generation did, i.e. something upstream of the
    estimators is wrong. On the real n=1000 run this identity showed a mean
    absolute residual of 0.19 nats, which is far too large for an identity.
    """
    from lm_polygraph.stat_calculators.greedy_probs import GreedyProbsCalculator

    model = _model(torch.float32)
    greedy = GreedyProbsCalculator(output_attentions=False, output_hidden_states=False)
    gen = greedy({}, PROMPTS, model, max_new_tokens=8)

    stats = _energies(model, PROMPTS, gen["greedy_tokens"])

    for i in range(len(PROMPTS)):
        n = len(gen["greedy_tokens"][i])
        tok = np.asarray(stats["energy_token_logits"][i], dtype=np.float64)
        lse = np.asarray(stats["energy_lse"][i][:n], dtype=np.float64)
        recovered = tok - lse  # = E^m - E^l = log p
        reference = np.asarray(gen["greedy_log_likelihoods"][i], dtype=np.float64)
        resid = np.abs(recovered - reference)
        assert (
            resid.max() < 1e-3
        ), f"sample {i}: log p = E^m - E^l violated by {resid.max():.5f} nats"
