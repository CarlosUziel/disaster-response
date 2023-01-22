import re
from typing import Iterable

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def is_binary(series: pd.Series) -> bool:
    """Check whether a pandas series contains only 0s and 1s.

    Args:
        series: Pandas series to check for 0s and 1s.

    Returns:
        Whether a series contains only 0s and 1s.
    """
    return sorted(series.dropna().unique()) == [0, 1]


def tokenize(text: str) -> Iterable[str]:
    """Pre-process text into lemmatized tokens.

    Args:
        text: Text to process.

    Returns:
        Iterable of lemmatized tokens.
    """
    # 0. Setup
    url_regex = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    non_alphanum_regex = r"[^a-z0-9]"

    # 1. Replace URLs
    text = re.sub(url_regex, " ", text)

    # 2. Replace non-alphanumeric characters
    text = re.sub(non_alphanum_regex, " ", text.lower())

    # 3. Tokenize and lemmatize text
    lemmatizer = WordNetLemmatizer()
    tokens = [
        lemmatizer.lemmatize(lemmatizer.lemmatize(token), pos="v")
        for token in word_tokenize(text)
        if token not in stopwords.words("english")
    ]

    return tokens
