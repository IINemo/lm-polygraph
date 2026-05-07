import numpy as np


def process_probs(
        probs: np.ndarray,
        min_p: float = 0.1,
        normalize_all: bool = True,
        B: int | None = None,
        best_variance: bool = False,
        debug: bool = False,
) -> np.ndarray:
    if best_variance:
        # no B, no min_p
        sorted_probs = sorted(list(probs), key=lambda x: -x)
        min_var, min_p = 1e9, -1
        all_vars = []
        for i in range(len(sorted_probs)):
            var_upper_bound = (1 - sum(sorted_probs[:i])) ** 2 / (len(sorted_probs) - i) / 4
            all_vars.append(var_upper_bound)
            if var_upper_bound < min_var:
                min_var = var_upper_bound
                min_p = (sorted_probs[i] + (sorted_probs[i - 1] if i > 0 else 2)) / 2
        if debug:
            print(f'selected min_p={min_p} with variance {min_var} (all variances: {all_vars})')
            print()
    elif B is not None:
        # B, no min_p
        min_p = ((probs[B] if B < len(probs) else -1) + (probs[B - 1] if B > 0 else 2)) / 2
    else:
        # no B, min_p
        assert min_p is not None

    if normalize_all:
        probs[probs < min_p] = min_p
        probs /= probs.sum()
        return probs
    else:
        large_probs = (probs >= min_p)
        large_probs_sum = probs[large_probs].sum()
        probs[~large_probs] = (1 - large_probs_sum) / (~large_probs).sum()
        return probs