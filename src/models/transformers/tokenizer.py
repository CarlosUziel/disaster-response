import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from data.utils import tokenize


class Tokenizer(BaseEstimator, TransformerMixin):
    """Transforms a text string into a list of tokens"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [tokenize(x) for x in np.array(X)]
