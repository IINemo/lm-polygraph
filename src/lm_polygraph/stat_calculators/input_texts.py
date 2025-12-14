from lm_polygraph.stat_calculators.stat_calculator import StatCalculator


class InputTextsCalculator(StatCalculator):
    def __init__(self):
        pass

    def __call__(
        self, batch_stats, inp_texts, model, max_new_tokens=None, *args, **kwargs
    ):
        return {"input_texts": inp_texts}

    @staticmethod
    def meta_info():
        return (["input_texts"], [])


def load_stat_calculator(config, environment):
    return InputTextsCalculator()
