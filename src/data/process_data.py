import logging
from pathlib import Path

import pandas as pd
import typer
from sqlalchemy import create_engine


def load_data(messages_filepath: Path, categories_filepath: Path) -> pd.DataFrame:
    """Load disaster datasets and merge.

    Args:
        messages_filepath: File path to messages dataset.
        categories_filepath: File path to categories dataset.

    Returns:
        merged disaster datasets.
    """
    logging.info(f"Loading data from {messages_filepath} and {categories_filepath}...")

    assert (
        messages_filepath.suffix == ".csv" and categories_filepath.suffix == ".csv"
    ), "Both dataset files must be .csv files"
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)

    logging.info("Data loaded.")

    return pd.merge(messages_df, categories_df)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean disaster data.

    Args:
        df: Merged disaster datasets.

    Returns:
        cleaned disaster data.
    """
    logging.info("Cleaning data...")
    # 1. Split message categories into separate columns
    categories_df = df["categories"].str.split(";", expand=True)

    # 2. Extract new column names
    categories_df.columns = [col[:-2] for col in categories_df.iloc[0]]

    # 3. Convert category values to 0s and 1s
    categories_df = categories_df.applymap(lambda x: int(x.split("-")[1]))

    # 4. Replace categories column with new category columns
    df = pd.concat([df.drop(columns=["categories"]), categories_df], axis=1)

    # 5. Remove duplicates
    df = df.drop_duplicates()

    logging.info("Data successfully cleaned.")

    return df


def save_data(df: pd.DataFrame, database_filepath: Path) -> None:
    """Save data in a sqlite database.

    Args:
        df: Disaster data.
        database_filepath: Path to save the database to.

    Returns:
        None
    """
    assert database_filepath.suffix == ".db", "Database filepath must end with .db"

    logging.info(f"Saving data to {database_filepath}...")

    engine = create_engine(f"sqlite:///{database_filepath}")
    df.to_sql("disaster_messages", engine, index=False)

    logging.info("Data successfully saved.")


def main(
    messages_filepath: Path = typer.Argument(
        Path(__file__).resolve().parents[2].joinpath("data/disaster/messages.csv"),
        help="File path to disaster messages dataset.",
    ),
    categories_filepath: Path = typer.Argument(
        Path(__file__).resolve().parents[2].joinpath("data/disaster/categories.csv"),
        help="File path to disaster categories dataset.",
    ),
    database_filepath: Path = typer.Argument(
        Path(__file__)
        .resolve()
        .parents[2]
        .joinpath("data/disaster/disaster_response.db"),
        help="File path to final database to store cleaned and pre-processed data in.",
    ),
):
    """ETL pipeline for disaster response data."""

    # 1. Load data
    df = load_data(messages_filepath, categories_filepath)

    # 2. Clean data
    df = clean_data(df)

    # 3. Save cleaned data
    save_data(df, database_filepath)


if __name__ == "__main__":
    typer.run(main)
