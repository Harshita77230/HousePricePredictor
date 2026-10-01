"""
app.py  —  Flask Web Frontend for House Price Predictor
Run:  python app.py   then open  http://127.0.0.1:5000
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Project root is always the directory that contains this file (app.py).
# Using Path(__file__).resolve() is safe regardless of CWD or how the module
# is imported (directly, via api/index.py, or by a Vercel serverless runtime).
_BASE_DIR = Path(__file__).resolve().parent

# Pin the template folder explicitly so Flask finds it even when this module
# is imported from a sub-package (e.g. api/index.py on Vercel).
app = Flask(__name__, template_folder=str(_BASE_DIR / "templates"))

MODEL_PATH = _BASE_DIR / "models" / "best_model.pkl"

# Load model once at startup — fail loudly so the error surfaces in logs
_data       = joblib.load(MODEL_PATH)
_pipeline   = _data["pipeline"]
_model_name = _data["model_name"]


def price_category(price):
    if price < 150_000:  return ("Budget",     "#6b7280")
    if price < 300_000:  return ("Affordable",  "#16a34a")
    if price < 500_000:  return ("Mid-Range",   "#2563eb")
    if price < 800_000:  return ("Premium",     "#9333ea")
    return                      ("Luxury",      "#dc2626")


@app.route("/")
def index():
    return render_template("index.html", model_name=_model_name)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        size_sqft = int(data["size_sqft"])
        bedrooms  = int(data["bedrooms"])
        bathrooms = int(data["bathrooms"])
        age_years = int(data["age_years"])
        garage    = int(data["garage"])
        location  = str(data["location"]).lower()

        # Validate
        assert 500  <= size_sqft <= 10000
        assert 1    <= bedrooms  <= 10
        assert 1    <= bathrooms <= 8
        assert 0    <= age_years <= 100
        assert 0    <= garage    <= 5
        assert location in ("urban", "suburban", "rural")

        sample = pd.DataFrame([{
            "size_sqft": size_sqft,
            "bedrooms":  bedrooms,
            "bathrooms": bathrooms,
            "age_years": age_years,
            "garage":    garage,
            "location":  location,
        }])

        price = float(_pipeline.predict(sample)[0])
        category, color = price_category(price)

        return jsonify({
            "price":      round(price),
            "formatted":  f"${price:,.0f}",
            "category":   category,
            "color":      color,
            "model_name": _model_name,
        })

    except (KeyError, ValueError, AssertionError) as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    print("\n  🏠  House Price Predictor — Web UI")
    print("  Open your browser at:  http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000)
