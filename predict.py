"""
predict.py  —  Interactive House Price Predictor
Enter house details and get an instant price estimate!
"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd

MODEL_PATH = "models/best_model.pkl"

# ── Helpers ────────────────────────────────────────────────────────────────────

def banner():
    print()
    print("=" * 55)
    print("   🏠  HOUSE PRICE PREDICTOR  🏠")
    print("=" * 55)
    print("  Enter your house details below.")
    print("  Type 'quit' at any prompt to exit.")
    print("=" * 55)


def get_int(prompt, min_val, max_val):
    while True:
        val = input(f"  {prompt} [{min_val}–{max_val}]: ").strip()
        if val.lower() == "quit":
            print("\n  Goodbye! 👋\n")
            sys.exit(0)
        try:
            v = int(val)
            if min_val <= v <= max_val:
                return v
            print(f"  ⚠  Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("  ⚠  Invalid input — please enter a whole number.")


def get_choice(prompt, choices):
    choices_str = " / ".join(f"{i+1}={c}" for i, c in enumerate(choices))
    while True:
        val = input(f"  {prompt} ({choices_str}): ").strip()
        if val.lower() == "quit":
            print("\n  Goodbye! 👋\n")
            sys.exit(0)
        # Accept number or name
        if val in [str(i+1) for i in range(len(choices))]:
            return choices[int(val) - 1]
        if val.lower() in [c.lower() for c in choices]:
            return val.lower()
        print(f"  ⚠  Please enter one of: {choices_str}")


def price_category(price):
    if price < 150_000:   return "Budget"
    if price < 300_000:   return "Affordable"
    if price < 500_000:   return "Mid-Range"
    if price < 800_000:   return "Premium"
    return "Luxury"


def show_result(inputs, price, model_name):
    print()
    print("─" * 55)
    print("  📋  YOUR HOUSE DETAILS")
    print("─" * 55)
    print(f"  Size        : {inputs['size_sqft']:,} sq ft")
    print(f"  Bedrooms    : {inputs['bedrooms']}")
    print(f"  Bathrooms   : {inputs['bathrooms']}")
    print(f"  House Age   : {inputs['age_years']} years")
    print(f"  Garage Spots: {inputs['garage']}")
    print(f"  Location    : {inputs['location'].capitalize()}")
    print("─" * 55)
    print(f"  💰  PREDICTED PRICE  :  ${price:,.0f}")
    print(f"  📊  CATEGORY         :  {price_category(price)}")
    print(f"  🤖  MODEL USED       :  {model_name}")
    print("─" * 55)


# ── Main Loop ──────────────────────────────────────────────────────────────────

def main():
    # Load model
    if not os.path.exists(MODEL_PATH):
        print("\n  ❌  Model not found!")
        print("  Please run  'python train_model.py'  first to train the model.\n")
        input("  Press Enter to exit...")
        sys.exit(1)

    data       = joblib.load(MODEL_PATH)
    pipeline   = data["pipeline"]
    model_name = data["model_name"]

    banner()
    print(f"\n  ✅  Model loaded: {model_name}\n")

    while True:
        # ── Collect inputs ─────────────────────────────────────────────────────
        print("\n  Enter house details:\n")

        size_sqft = get_int("House size (sq ft)",      500, 10000)
        bedrooms  = get_int("Number of bedrooms",        1,     10)
        bathrooms = get_int("Number of bathrooms",       1,      8)
        age_years = get_int("Age of house (years)",      0,    100)
        garage    = get_int("Garage spots (0 = none)",   0,      5)
        location  = get_choice("Location type", ["urban", "suburban", "rural"])

        # ── Predict ────────────────────────────────────────────────────────────
        inputs = {
            "size_sqft": size_sqft,
            "bedrooms":  bedrooms,
            "bathrooms": bathrooms,
            "age_years": age_years,
            "garage":    garage,
            "location":  location,
        }
        sample = pd.DataFrame([inputs])
        price  = pipeline.predict(sample)[0]

        show_result(inputs, price, model_name)

        # ── Ask again ──────────────────────────────────────────────────────────
        print()
        again = input("  Would you like to predict another house? (yes/no): ").strip().lower()
        if again not in ("yes", "y"):
            print("\n  Thanks for using House Price Predictor! 👋\n")
            break


if __name__ == "__main__":
    main()
