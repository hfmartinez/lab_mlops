from flask import Flask, request, render_template, jsonify
from pycaret.regression import load_model, predict_model
import pandas as pd

app = Flask(__name__)

model = load_model("model")
cols = ["age", "sex", "bmi", "children", "smoker", "region"]


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    values = [x for x in request.form.values()]
    data_unseen = pd.DataFrame([values], columns=cols)

    prediction_df = predict_model(model, data=data_unseen, round=2)
    prediction = float(prediction_df["prediction_label"].iloc[0])

    return render_template("home.html", pred=f"Expected Bill will be {prediction}")


@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data], columns=cols)

    prediction_df = predict_model(model, data=data_unseen, round=2)
    output = float(prediction_df["prediction_label"].iloc[0])

    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
