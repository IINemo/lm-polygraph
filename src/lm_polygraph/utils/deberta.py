import torch

from transformers import (
    DebertaForSequenceClassification,
    DebertaTokenizer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


class Deberta:
    """
    Allows for the implementation of a singleton DeBERTa model which can be shared across
    different uncertainty estimation methods in the code.
    """

    def __init__(
        self,
        deberta_path: str = "microsoft/deberta-large-mnli",
        batch_size: int = 10,
        device=None,
    ):
        """
        Parameters
        ----------
        deberta_path : str
            huggingface path of the pretrained DeBERTa (default 'microsoft/deberta-large-mnli')
        device : str
            device on which the computations will take place (default 'cuda:0' if available, else 'cpu').
        """
        self.deberta_path = deberta_path
        self.batch_size = batch_size
        self._deberta = None
        self._deberta_tokenizer = None
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.setup()

    @property
    def deberta(self):
        if self._deberta is None:
            self.setup()

        return self._deberta

    @property
    def deberta_tokenizer(self):
        if self._deberta_tokenizer is None:
            self.setup()

        return self._deberta_tokenizer

    def to(self, device):
        self.device = device
        if self._deberta is not None:
            self._deberta.to(self.device)

    def setup(self):
        """
        Loads and prepares the DeBERTa model from the specified path.
        """
        if self._deberta is not None:
            return
        self._deberta = DebertaForSequenceClassification.from_pretrained(
            self.deberta_path, problem_type="multi_label_classification"
        )
        self._deberta_tokenizer = DebertaTokenizer.from_pretrained(self.deberta_path)
        self._deberta.to(self.device)
        self._deberta.eval()


class MultilingualDeberta(Deberta):
    """
    Allows for the implementation of a singleton multilingual DeBERTa model which can be shared across
    different uncertainty estimation methods in the code.
    """

    def __init__(
        self,
        deberta_path: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        batch_size: int = 10,
        device=None,
    ):
        """
        Parameters
        ----------
        deberta_path : str
            huggingface path of the pretrained DeBERTa (default
            'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7')
        device : str
            device on which the computations will take place (default 'cuda:0' if available, else 'cpu').
        """
        self.deberta_path = deberta_path
        self.batch_size = batch_size
        self._deberta = None
        self._deberta_tokenizer = None
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.setup()

    def setup(self):
        """
        Loads and prepares the DeBERTa model from the specified path.
        """
        if self._deberta is not None:
            return
        self._deberta_tokenizer = AutoTokenizer.from_pretrained(self.deberta_path)
        self._deberta = AutoModelForSequenceClassification.from_pretrained(
            self.deberta_path
        )
        self._deberta.to(self.device)
        self._deberta.eval()
        # Make label2id classes uppercase to match implementation of microsoft/deberta-large-mnli
        self._deberta.deberta.config.label2id = {
            k.upper(): v for k, v in self._deberta.deberta.config.label2id.items()
        }
