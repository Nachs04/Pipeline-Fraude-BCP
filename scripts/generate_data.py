"""Generador de datos sintéticos para pruebas del pipeline."""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 50000

dates = pd.date_range("2026-01-01", periods=n, freq="min")

df = pd.DataFrame({
    "timestamp": dates,
    "customer_id": rng.integers(1, 3001, n).astype(str),
    "terminal_id": rng.integers(1, 501, n).astype(str),
    "amount": np.round(rng.lognormal(mean=3.5, sigma=0.8, size=n), 2),
})

# Etiqueta sintética, solo para el experimento académico.
risk = (
    (df["amount"] > df["amount"].quantile(0.985)).astype(int)
    + (rng.random(n) < 0.008).astype(int)
)
df["fraud"] = (risk >= 2).astype(int)

Path("data").mkdir(exist_ok=True)
df.to_csv("data/transactions_50000.csv", index=False)
print("Generado data/transactions_50000.csv")
