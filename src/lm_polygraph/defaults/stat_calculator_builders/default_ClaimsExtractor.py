from lm_polygraph.stat_calculators.extract_claims import ClaimsExtractor
from lm_polygraph.utils.openai_chat import OpenAIChat


def load_stat_calculator(config, builder):
    if not hasattr(builder, "chat_model"):
        builder.chat_model = OpenAIChat(
            openai_model=config.openai_model,
            base_url=getattr(config, "base_url", None),
            timeout=getattr(config, "timeout", 600),
            cache_path=config.cache_path,
        )

    return ClaimsExtractor(
        builder.chat_model,
        language=config.language,
        n_threads=getattr(config, "n_threads", 1),
    )
