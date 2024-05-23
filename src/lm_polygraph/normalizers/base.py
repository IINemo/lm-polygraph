# Description: Base class for all UE normalizers


class BaseUENormalizer:
    def __init__(self):
        pass

    def fit(self, gen_metrics, ues):
        raise NotImplementedError("fit method not implemented")

    def transform(self, ues):
        raise NotImplementedError("transform method not implemented")
