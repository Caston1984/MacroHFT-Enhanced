import numpy as np
import pandas as pd

def generate_mock_lob_data(num_rows=1000):
    """Generates synthetic Limit Order Book (LOB) data matching the expected schema 
    for feature engineering validation without exposing proprietary data sources."""
    np.random.seed(42)
    timestamps = pd.date_range(start="2026-01-01", periods=num_rows, freq="100ms")
    
    data = {
        "timestamp": timestamps,
        "bid_price_1": np.linspace(100.0, 101.0, num_rows) + np.random.normal(0, 0.01, num_rows),
        "ask_price_1": np.linspace(100.02, 101.02, num_rows) + np.random.normal(0, 0.01, num_rows),
        "bid_volume_1": np.random.randint(100, 1000, num_rows),
        "ask_volume_1": np.random.randint(100, 1000, num_rows),
    }
    df = pd.DataFrame(data)
    df.to_csv("data/processed/sample_lob_data.csv", index=False)
    print("Mock LOB dataset generated successfully at data/processed/sample_lob_data.csv")

if __name__ == "__main__":
    generate_mock_lob_data()
