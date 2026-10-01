"""
generate_dataset.py  —  Creates synthetic house price dataset
"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

RANDOM_SEED = 42
NUM_SAMPLES = 1000

np.random.seed(RANDOM_SEED)


def generate_dataset(n: int = NUM_SAMPLES) -> pd.DataFrame:
    size_sqft = np.random.randint(500, 5000, n)
    bedrooms  = np.random.randint(1, 7, n)
    bathrooms = np.clip(bedrooms - np.random.randint(0, 2, n), 1, 6)
    age_years = np.random.randint(0, 50, n)
    garage    = np.random.randint(0, 4, n)
    location  = np.random.choice(["urban", "suburban", "rural"], n, p=[0.4, 0.4, 0.2])

    loc_map  = {"urban": 1.4, "suburban": 1.0, "rural": 0.7}
    loc_mult = np.array([loc_map[l] for l in location])

    base_price = (
        50_000
        + size_sqft * 120
        + bedrooms  * 8_000
        + bathrooms * 6_000
        - age_years * 500
        + garage    * 5_000
    ) * loc_mult

    noise = np.random.normal(0, 15_000, n)
    price = np.clip(base_price + noise, 50_000, None).astype(int)

    return pd.DataFrame({
        "size_sqft": size_sqft, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "age_years": age_years, "garage": garage, "location": location, "price": price,
    })


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset()
    df.to_csv("data/houses.csv", index=False)
    print(f"Dataset saved → data/houses.csv  ({df.shape[0]} rows)")
