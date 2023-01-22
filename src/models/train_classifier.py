import logging
import multiprocessing
import pickle
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import typer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

from data.utils import is_binary, tokenize
from models.utils import get_classification_metrics, multi_label_balanced_accuracy

logging.basicConfig(force=True)
logging.getLogger().setLevel(logging.INFO)


def load_data(database_filepath: Path) -> Tuple[np.array, np.array, Iterable[str]]:
    """Load data from database for modeling.

    Args:
        database_filepath: Path to save the database to.

    Returns:
        Input features.
        Target values.
        Category names.
    """
    # 1. Load data
    assert database_filepath.suffix == ".db", "Database filepath must end with .db"
    logging.info(f"Loading data from {database_filepath}...")

    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql("disaster_messages", engine)

    logging.info(f"Data loaded successfully!")

    # 2. Separate data into input features and target values
    X = df["message"]

    category_columns = [col for col in df.columns if is_binary(df[col])]
    y = df[category_columns]

    return np.ascontiguousarray(X), np.ascontiguousarray(y), category_columns


def build_pipeline(random_seed: int = 8080) -> Pipeline:
    """Build machine learning pipeline to be used to model the data.

    Args:
        random_seed: Seed to initialize random state.

    Returns:
        Scikit-learn pipeline estimator.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    tokenizer=tokenize,
                    token_pattern=None,
                    strip_accents="unicode",
                ),
            ),
            ("clf", RandomForestClassifier(random_state=random_seed)),
        ]
    )


def tune_evaluate_model(
    model: Pipeline,
    X: np.array,
    y: np.array,
    category_names: Iterable[str],
    n_splits: int = 5,
    n_jobs: int = 4,
    random_seed: int = 8080,
) -> Tuple[Pipeline, pd.DataFrame]:
    """Evaluate a model on test data after finding the best hyper-parameters.
    A more computationally-expensive but also stricter option would use two K-fold loops
    instead, where the model is tuned and tested on each fold.

    The model solves a multi-label classification problem.

    Args:
        model: Model to train.
        X: Data input features.
        y: Data target values.
        category_names: Name of multi-label categories.
        n_splits: Number of cross K-cross validation splits.
        n_jobs: Number of jobs to run in parallel.
        random_state: Seed to initialize random state.

    Returns:
        Model refitted to all data using optimal hyper-parameters.
        Grid Search results.

    """
    # 0. Setup
    hparams = {
        "clf__n_estimators": [5, 10, 25, 50],
        "clf__criterion": ["gini", "entropy"],
        "clf__max_samples": [0.3, 0.6, 1],
        "clf__max_depth": [4, 8, 16],
        "clf__max_features": ["sqrt", "log2"],
        "clf__class_weight": ["balanced", "balanced_subsample"],
    }

    # 1. Build GridSearch estimator
    gridsearch_estimator = GridSearchCV(
        deepcopy(model),
        hparams,
        scoring=make_scorer(multi_label_balanced_accuracy),
        cv=n_splits,
        n_jobs=n_jobs,
        verbose=2,
    )

    # 2. Split data into train and validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, shuffle=True
    )

    # 3. Tune model hyper-parameters on train data
    logging.info("Tuning model using Grid Search...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gridsearch_estimator.fit(X_train, y_train)

    logging.info("Tuning successfully finished!")

    # 4. Gather train and test metrics
    logging.info("Gathering performance metrics...")

    performance_metrics = {
        (category_name, m_set): get_classification_metrics(
            ground_truth[:, i], predicted[:, i]
        )
        for m_set, ground_truth, predicted in zip(
            ("train", "test"),
            (y_train, y_test),
            (
                gridsearch_estimator.predict(X_train),
                gridsearch_estimator.predict(X_test),
            ),
        )
        for i, category_name in enumerate(category_names)
    }

    logging.info("Performance metrics successfully computed!")

    # 5. Retrain on all data using best tuned parameters
    logging.info("Retraining on all data using optimal hyper-parameters...")

    final_model = deepcopy(model)
    final_model.set_params(**gridsearch_estimator.best_params_)
    final_model.fit(X, y)

    logging.info("Final model ready for deployment!")

    return final_model, pd.DataFrame(performance_metrics).transpose().unstack(level=1)


def save_model(
    model: Pipeline,
    model_filepath: Path,
    performance_metrics: Optional[pd.DataFrame] = None,
    performance_metrics_filepath: Optional[Path] = None,
):
    """Save model to disk.

    Args:
        model: Scikit-learn model to save to disk.
        model_filepath: Path to save model to.
        performance_metrics: Model performance metrics.
        performance_metrics_filepath: Path to store performance metrics to.
    """
    # 1. Save model to disk
    assert model_filepath.suffix == ".pkl"
    logging.info("Saving model to disk...")

    pickle.dump(model, model_filepath.open("wb"))

    logging.info("Model saved successfully!")

    # 2. Save K-fold test metrics to disk
    assert performance_metrics_filepath.suffix == ".csv"
    logging.info("Saving performance metrics to disk...")

    performance_metrics.to_csv(performance_metrics_filepath)

    logging.info("Performance metrics saved successfully!")


def main(
    database_filepath: Path = typer.Argument(
        Path(__file__)
        .resolve()
        .parents[2]
        .joinpath("data/disaster/disaster_response.db"),
        help="File path to sqlite database containing input data.",
    ),
    model_filepath: Path = typer.Argument(
        Path(__file__)
        .resolve()
        .parents[2]
        .joinpath("data/models/disaster_response.pkl"),
        help="File path to store final model in.",
    ),
    performance_metrics_filepath: Path = typer.Argument(
        Path(__file__)
        .resolve()
        .parents[2]
        .joinpath("data/models/train_test_metrics.csv"),
        help="File path to store model performance metrics.",
    ),
    random_seed: int = typer.Option(
        8080,
        help="Seed to initialize random state.",
    ),
    n_splits: int = typer.Option(
        5,
        help="Number of K-fold cross-validation splits.",
    ),
    n_jobs: int = typer.Option(
        multiprocessing.cpu_count() // 2,
        help=(
            "Number of jobs to run in parallel for Grid Search hyper-parameter tuning."
        ),
    ),
):
    """Machine Learning pipeline for disaster response data."""
    # 1. Load data
    X, y, category_names = load_data(database_filepath)

    # 2. Build model pipeline
    model = build_pipeline(random_seed=random_seed)

    # 3. Evaluate and tune model
    final_model, performance_metrics = tune_evaluate_model(
        model,
        X,
        y,
        category_names,
        n_splits=n_splits,
        n_jobs=n_jobs,
        random_seed=random_seed,
    )

    # 4. Save model to disk
    save_model(
        final_model,
        model_filepath,
        performance_metrics=performance_metrics,
        performance_metrics_filepath=performance_metrics_filepath,
    )


if __name__ == "__main__":
    typer.run(main)
