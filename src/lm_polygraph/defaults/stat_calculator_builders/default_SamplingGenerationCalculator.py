import logging
from lm_polygraph.stat_calculators.sample import SamplingGenerationCalculator

log = logging.getLogger("lm_polygraph")


def load_stat_calculator(config, builder):
    samples_n = getattr(config, "samples_n", 10)
    temperature = getattr(config, "temperature", None)
    log.info(f'Configured SamplingGenerationCalculator with {samples_n} samples and T={temperature}')
    return SamplingGenerationCalculator(samples_n=samples_n, temperature=temperature)
