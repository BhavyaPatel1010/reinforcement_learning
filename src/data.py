"""Dataset loading and column inference for common Kaggle retail files."""
from __future__ import annotations
import numpy as np
import pandas as pd


DEFAULT_KAGGLE_DATASET = "atomicd/retail-store-inventory-and-demand-forecasting"


def download_kaggle_dataset(handle: str = DEFAULT_KAGGLE_DATASET) -> str:
    """Download a Kaggle dataset and return its largest CSV file.

    KaggleHub is already authenticated inside Kaggle notebooks.  Selecting the
    largest CSV avoids accidentally choosing a small metadata/sample file when
    a dataset version contains more than one CSV.
    """
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Install KaggleHub first: pip install kagglehub") from exc

    from pathlib import Path

    dataset_dir = Path(kagglehub.dataset_download(handle))
    csv_files = list(dataset_dir.rglob("*.csv"))
    if not csv_files:
        available = [str(p.relative_to(dataset_dir)) for p in dataset_dir.rglob("*") if p.is_file()]
        raise FileNotFoundError(
            f"Dataset {handle!r} downloaded to {dataset_dir}, but it contains no CSV. "
            f"Files found: {available[:20]}"
        )
    selected = max(csv_files, key=lambda path: path.stat().st_size)
    print(f"Downloaded Kaggle dataset: {handle}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Selected CSV: {selected}")
    return str(selected)


def synthetic_data(days=900, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    promo = (rng.random(days) < 0.16).astype(float)
    price = 12 + 0.8 * np.sin(t / 30) - 1.5 * promo
    demand = np.maximum(0, 18 + 7 * np.sin(2 * np.pi * t / 7) + 5 * np.sin(t / 38) + 8 * promo + rng.normal(0, 4, days))
    return pd.DataFrame({"demand": demand, "price": price, "promotion": promo})


def load_retail_data(path: str | None, product_id: str | None = None, store_id: str | None = None):
    if not path:
        return synthetic_data(), "synthetic demonstration data"
    df = pd.read_csv(path)
    normalized = {c.lower().replace(" ", "_"): c for c in df.columns}
    def find(options, required=False):
        for name in options:
            if name in normalized: return normalized[name]
        if required: raise ValueError(f"Could not find one of {options}. CSV columns: {list(df.columns)}")
        return None
    date = find(["date", "day"])
    demand = find(["demand", "units_sold", "sales", "demanda_uni_equil"], required=True)
    product = find(["product_id", "item", "item_id", "producto_id"])
    store = find(["store_id", "store", "store_nbr", "agencia_id"])
    price = find(["price", "sell_price"])
    promotion = find(["promotion", "onpromotion", "discount"])
    if product and product_id is not None: df = df[df[product].astype(str) == str(product_id)]
    elif product: df = df[df[product] == df[product].iloc[0]]
    if store and store_id is not None: df = df[df[store].astype(str) == str(store_id)]
    elif store: df = df[df[store] == df[store].iloc[0]]
    if df.empty: raise ValueError("No rows remained after product/store selection.")
    if date: df = df.sort_values(date)
    result = pd.DataFrame({"demand": pd.to_numeric(df[demand], errors="coerce").fillna(0)})
    result["price"] = pd.to_numeric(df[price], errors="coerce").fillna(1.0) if price else 1.0
    result["promotion"] = pd.to_numeric(df[promotion], errors="coerce").fillna(0.0) if promotion else 0.0
    result = result.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return result, f"{len(result)} rows from {path}"
