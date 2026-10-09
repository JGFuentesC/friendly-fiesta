"""Genera los archivos de train y validation a partir del dataset crudo.

División aleatoria 70/30 con semilla fija (seed=42) para que el resultado sea
reproducible. El dataset crudo se lee de data/raw/file.csv y los archivos de
salida se escriben en data/processed/train.csv y data/processed/validation.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
TRAIN_FRAC = 0.70

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "data" / "raw" / "file.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

COLUMNS = ["x1", "x2", "x3", "y"]


def main() -> None:
    df = pd.read_csv(RAW_PATH, sep=r"\s+")
    df.columns = COLUMNS

    rng = np.random.default_rng(SEED)
    n = len(df)
    n_train = int(np.floor(TRAIN_FRAC * n))
    idx = rng.permutation(n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    train = df.iloc[train_idx].reset_index(drop=True)
    val = df.iloc[val_idx].reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "validation.csv", index=False)

    print(f"dataset: {n} filas")
    print(f"train: {len(train)} filas ({len(train) / n:.2%})")
    print(f"validation: {len(val)} filas ({len(val) / n:.2%})")
    print(f"escritos en {PROCESSED_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
