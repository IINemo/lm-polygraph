from lm_polygraph.stat_calculators.attention import (
    LookbackRatioCalculator,
    AttentionFeaturesCalculator,
)
import logging

log = logging.getLogger("lm_polygraph")


def load_stat_calculator(config, builder):
    if config.attentions == "lookback_ratio":
        builder.return_lookback_ratios = True
        return LookbackRatioCalculator()
    elif config.attentions == "attention_features":
        builder.return_attention_features = True
        return AttentionFeaturesCalculator()
    else:
        raise ValueError("Invalid attentions configuration")
