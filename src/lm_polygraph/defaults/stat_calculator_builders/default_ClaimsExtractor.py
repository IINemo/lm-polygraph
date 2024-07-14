from lm_polygraph.stat_calculators.extract_claims import ClaimsExtractor
from lm_polygraph.utils.openai_chat import SingletonOpenAIChat


def load_stat_calculator(config, builder):
    chat_model = SingletonOpenAIChat(config.openai_model, config.cache_path)
    return ClaimsExtractor(chat_model)
