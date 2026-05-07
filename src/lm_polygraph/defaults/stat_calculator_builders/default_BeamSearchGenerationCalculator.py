import logging
from lm_polygraph.stat_calculators.beamsearch import BeamSearchGenerationCalculator

log = logging.getLogger("lm_polygraph")


def load_stat_calculator(config, builder):
    beams_n = getattr(config, "beams_n", 10)
    num_beam_groups = getattr(config, "num_beam_groups", None)
    diversity_penalty = getattr(config, "diversity_penalty", None)
    log.info(f'Configured BeamSearchGenerationCalculator with {beams_n} beams')
    return BeamSearchGenerationCalculator(
        beams_n=beams_n,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
    )
