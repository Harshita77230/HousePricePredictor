# 🏠 House Price Predictor

A machine learning web application that predicts house prices based on key property features.

## Live Demo

Deployed on Vercel: [HousePricePredictor](https://github.com/Harshita77230/HousePricePredictor)

## Features

- Predicts house prices using a trained **Gradient Boosting** model
- Web UI built with Flask + HTML/CSS/JavaScript
- Instant predictions via REST API (`/predict`)
- Price category labels: Budget → Affordable → Mid-Range → Premium → Luxury

## Input Features

| Feature | Description | Range |
|---|---|---|
| `size_sqft` | House size in square feet | 500 – 10,000 |
| `bedrooms` | Number of bedrooms | 1 – 10 |
| `bathrooms` | Number of bathrooms | 1 – 8 |
| `age_years` | Age of the house in years | 0 – 100 |
| `garage` | Number of garage spots | 0 – 5 |
| `location` | Location type | urban / suburban / rural |

## Tech Stack

- **Python** — core language
- **Scikit-learn** — ML pipeline (GradientBoostingRegressor, StandardScaler, OneHotEncoder)
- **Pandas / NumPy** — data handling
- **Joblib** — model serialization
- **Flask** — web framework
- **Vercel** — serverless deployment

## Local Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (generates data + saves models/best_model.pkl)
python train_model.py

# 3. Run the web server
python app.py
# Open http://127.0.0.1:5000
```

Or use the provided batch files on Windows:
1. `1_SETUP_AND_TRAIN.bat` — installs packages and trains the model
2. `2_PREDICT.bat` — terminal prediction app
3. `3_RUN_WEB.bat` — starts the Flask web UI

## Project Structure

```
HousePricePredictor/
├── api/
│   └── index.py          # Vercel WSGI entry point
├── models/
│   └── best_model.pkl    # Trained model (committed)
├── templates/
│   └── index.html        # Web UI
├── app.py                # Flask application
├── train_model.py        # Model training script
├── generate_dataset.py   # Synthetic dataset generator
├── predict.py            # Terminal prediction app
├── requirements.txt      # Runtime dependencies
└── vercel.json           # Vercel deployment config
```
