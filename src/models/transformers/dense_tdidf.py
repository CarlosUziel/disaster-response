import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class DenseTfidfVectorizer(TfidfVectorizer):
    """Transforms TfidfVectorizer output to dense matrix."""

    def fit_transform(self, *args, **kwargs):
        return np.ascontiguousarray(super().fit_transform(*args, **kwargs).toarray())

    def transform(self, *args, **kwargs):
        return np.ascontiguousarray(super().transform(*args, **kwargs).toarray())
