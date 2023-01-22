import nltk
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from data.utils import tokenize


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """Feature extractor. Determines whether any sentence in a text start with a verb."""

    def starting_verb(self, text: str):
        """Determines whether any sentence in a text start with a verb.

        Args:
            text: Text to analyse.
        """
        for sentence in nltk.sent_tokenize(text):
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags) >= 1:
                first_word, first_tag = pos_tags[0]
                if first_tag in ["VB", "VBP"] or first_word == "RT":
                    return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self.starting_verb(x) for x in np.array(X)])[..., None]
