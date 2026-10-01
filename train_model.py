"""
train_model.py  —  Trains the ML model and saves it to models/best_model.pkl
Run this ONCE before using predict.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from generate_dataset import generate_dataset

DATA_PATH  = "data/houses.csv"
OUTPUT_DIR = "outputs"
MODEL_DIR  = "models"
os.makedirs("data",     exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

NUMERIC_FEATURES     = ["size_sqft", "bedrooms", "bathrooms", "age_years", "garage"]
CATEGORICAL_FEATURES = ["location"]
TARGET               = "price"


def load_data():
    if not os.path.exists(DATA_PATH):
        print("Generating dataset...")
        df = generate_dataset()
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)
    print(f"Dataset: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def build_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("scaler", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_FEATURES),
    ])


def train_and_save():
    df = load_data()
    X  = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y  = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pre = build_preprocessor()
    models = {
        "Linear Regression":  Pipeline([("pre", pre), ("model", LinearRegression())]),
        "Ridge Regression":   Pipeline([("pre", pre), ("model", Ridge(alpha=10))]),
        "Random Forest":      Pipeline([("pre", pre), ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))]),
        "Gradient Boosting":  Pipeline([("pre", pre), ("model", GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=4, random_state=42))]),
    }

    print("\n" + "="*55)
    print("  MODEL TRAINING & EVALUATION")
    print("="*55)

    results = []
    trained = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        cv   = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2", n_jobs=-1).mean()
        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2_test": r2, "R2_cv": cv})
        trained[name] = pipe
        print(f"  {name:<22}  R²={r2:.4f}  MAE=${mae:>9,.0f}  RMSE=${rmse:>9,.0f}")

    df_res      = pd.DataFrame(results).sort_values("R2_test", ascending=False)
    best_name   = df_res.iloc[0]["Model"]
    best_pipe   = trained[best_name]

    print(f"\n  Best Model : {best_name}")
    print(f"  R² Score   : {df_res.iloc[0]['R2_test']:.4f}")
    print(f"  MAE        : ${df_res.iloc[0]['MAE']:,.0f}")
    print(f"  RMSE       : ${df_res.iloc[0]['RMSE']:,.0f}")

    # Save model
    model_path = f"{MODEL_DIR}/best_model.pkl"
    joblib.dump({"pipeline": best_pipe, "model_name": best_name}, model_path)
    print(f"\n  Model saved → {model_path}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Model Evaluation", fontsize=14, fontweight="bold")

    # R² bar
    names  = df_res["Model"].tolist()
    r2s    = df_res["R2_test"].tolist()
    colors = ["#3b82d4" if i == 0 else "#b0c4de" for i in range(len(names))]
    axes[0].barh(names, r2s, color=colors)
    axes[0].set_xlabel("R² Score"); axes[0].set_title("R² Comparison"); axes[0].set_xlim(0, 1.05)
    for i, v in enumerate(r2s):
        axes[0].text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=9)

    # RMSE bar
    rmses = df_res["RMSE"].tolist()
    axes[1].barh(names, [r/1000 for r in rmses], color=colors)
    axes[1].set_xlabel("RMSE ($000s)"); axes[1].set_title("RMSE (lower=better)")
    for i, v in enumerate(rmses):
        axes[1].text(v/1000 + 0.3, i, f"${v/1000:.1f}k", va="center", fontsize=9)

    # Actual vs Predicted
    y_pred_best = best_pipe.predict(X_test)
    axes[2].scatter(y_test/1000, y_pred_best/1000, alpha=0.4, s=15, color="#3b82d4")
    mn, mx = y_test.min()/1000, y_test.max()/1000
    axes[2].plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
    axes[2].set_xlabel("Actual ($000s)"); axes[2].set_ylabel("Predicted ($000s)")
    axes[2].set_title(f"Actual vs Predicted\n({best_name})"); axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()

    # Feature importance
    model_step = best_pipe.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        ohe_feats = list(best_pipe.named_steps["pre"].named_transformers_["cat"]
                         .named_steps["ohe"].get_feature_names_out(CATEGORICAL_FEATURES))
        feat_names = NUMERIC_FEATURES + ohe_feats
        fi_df = pd.DataFrame({"Feature": feat_names, "Importance": model_step.feature_importances_})\
                  .sort_values("Importance", ascending=True)
        plt.figure(figsize=(9, 5))
        plt.barh(fi_df["Feature"], fi_df["Importance"], color="#3b82d4")
        plt.xlabel("Importance"); plt.title("Feature Importance"); plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=130, bbox_inches="tight")
        plt.close()

    print(f"  Plots saved → {OUTPUT_DIR}/")
    print("\n  Training complete! Now run:  python predict.py")


if __name__ == "__main__":
    train_and_save()
