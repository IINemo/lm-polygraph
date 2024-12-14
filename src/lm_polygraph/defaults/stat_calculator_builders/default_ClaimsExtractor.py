from lm_polygraph.stat_calculators.extract_claims import ClaimsExtractor
from lm_polygraph.utils.openai_chat import OpenAIChat


def load_stat_calculator(config, builder):
    if not hasattr(builder, "chat_model"):
        builder.chat_model = OpenAIChat(config.openai_model, config.cache_path)

    return ClaimsExtractor(builder.chat_model, language=config.language)
