import logging
import multiprocessing
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import typer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

from data.utils import is_binary, tokenize
from models.transformers.dense_tdidf import DenseTfidfVectorizer
from models.utils import get_best_cv_indx, get_classification_metrics

logging.basicConfig()
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
                DenseTfidfVectorizer(
                    tokenizer=tokenize,
                    token_pattern=None,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                RandomForestClassifier(random_state=random_seed),
            ),
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
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
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
        Model test performance metrics.
        Grid Search results.

    """
    # 0. Setup
    hparams = {
        "tfidf__sublinear_tf": [True, False],
        "tfidf__min_df": (0.01, 0.05, 0.1),
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [None, 16, 32, 64],
        "clf__class_weight": ["balanced", "balanced_subsample"],
    }

    # 1. Build GridSearch estimator
    gridsearch_estimator = GridSearchCV(
        deepcopy(model),
        hparams,
        scoring=["f1_weighted", "precision_weighted", "recall_weighted"],
        refit=get_best_cv_indx,
        cv=n_splits,
        n_jobs=n_jobs,
        verbose=2,
    )

    # 2. Split data into train and validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        # make it big to increase chances of getting samples of all labels
        test_size=0.3,
        random_state=random_seed,
        shuffle=True,
    )

    train_dist = {cat: y_train[:, i].sum() for i, cat in enumerate(category_names)}
    logging.info(f"Labels distribution in training samples:\n{train_dist}")
    test_dist = {cat: y_test[:, i].sum() for i, cat in enumerate(category_names)}
    logging.info(f"Labels distribution in testing samples:\n{test_dist}")

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

    return (
        final_model,
        pd.DataFrame(performance_metrics).transpose().unstack(level=1),
        pd.DataFrame(gridsearch_estimator.cv_results_).sort_values(
            [
                "mean_test_f1_weighted",
                "mean_test_precision_weighted",
                "mean_test_recall_weighted",
            ],
            ascending=False,
        ),
    )


def save_results(
    model: Pipeline,
    results_path: Path,
    performance_metrics: Optional[pd.DataFrame] = None,
    cv_results: Optional[pd.DataFrame] = None,
):
    """Save model to disk.

    Args:
        model: Scikit-learn model to save to disk.
        results_path: Where to store results.
        performance_metrics: Model performance metrics.
        cv_results: Grid Search CV results.
    """
    # 1. Save model to disk
    logging.info("Saving model to disk...")

    joblib.dump(model, results_path.joinpath("model.pkl").open("wb"), compress=9)

    logging.info("Model saved successfully!")

    # 2. Save model metrics to disk
    if performance_metrics is not None:
        logging.info("Saving performance metrics to disk...")

        performance_metrics.to_csv(results_path.joinpath("performance_metrics.csv"))

        logging.info("Performance metrics saved successfully!")

    # 3. Save Grid Search CV results
    if cv_results is not None:
        logging.info("Grid Search CV results to disk...")

        cv_results.to_csv(results_path.joinpath("cv_results.csv"))

        logging.info("Grid Search CV results saved successfully!")


def main(
    database_filepath: Path = typer.Argument(
        Path(__file__)
        .resolve()
        .parents[2]
        .joinpath("data/disaster/disaster_response.db"),
        help="File path to sqlite database containing input data.",
    ),
    results_path: Path = typer.Argument(
        Path(__file__).resolve().parents[2].joinpath("data/models"),
        help="Directory to save results to, including model and metrics.",
    ),
    random_seed: int = typer.Option(
        8080,
        help="Seed to initialize random state.",
    ),
    n_splits: int = typer.Option(
        4,
        help="Number of K-fold cross-validation splits.",
    ),
    n_jobs: int = typer.Option(
        multiprocessing.cpu_count() - 2,
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
    final_model, performance_metrics, cv_results = tune_evaluate_model(
        model,
        X,
        y,
        category_names,
        n_splits=n_splits,
        n_jobs=n_jobs,
        random_seed=random_seed,
    )

    # 4. Save model to disk
    save_results(
        final_model,
        results_path,
        performance_metrics=performance_metrics,
        cv_results=cv_results,
    )


if __name__ == "__main__":
    typer.run(main)
