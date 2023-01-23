import json
from pathlib import Path

import joblib
import pandas as pd
import plotly
from flask import Flask, render_template, request
from plotly.graph_objs import Bar, Pie
from sqlalchemy import create_engine

app = Flask(__name__)


# load data
data_path = Path(__file__).parents[2].joinpath("data")
db_path = data_path.joinpath("disaster/disaster_response.db")
engine = create_engine(f"sqlite:///{db_path}")
df = pd.read_sql("disaster_messages", engine)

# load model
model = joblib.load(data_path.joinpath("models/model.joblib"))


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():
    # extract data needed for visuals
    genre_counts = df.groupby("genre").count()["message"]

    # category counts
    category_counts = (
        df[[col for col in df.columns if sorted(df[col].dropna().unique()) == [0, 1]]]
        .sum()
        .sort_values(ascending=False)
    )

    # create visuals
    graphs = [
        {
            "data": [Bar(x=genre_counts.index.tolist(), y=genre_counts)],
            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Genre"},
            },
        },
        {
            "data": [
                Pie(labels=category_counts.index.tolist(), values=category_counts, hole=.3)
            ],
            "layout": {
                "title": "Distribution of Message Categories in data",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Category"},
            },
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
