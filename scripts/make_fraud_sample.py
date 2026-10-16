"""Genera submuestreos del dataset de fraude para el entrenamiento en vivo.

Salidas (gitignored por `data/processed/fraud*`):
- data/processed/fraud_train.csv   ~100k filas, fraude sobremuestreado ~10 %
- data/processed/fraud_test.csv    ~300k filas con la tasa real de fraude (0.12 %)

El train sobremuestrea la clase minoritaria para que la red la aprenda; el test
conserva la tasa real para que el FPR sea honesto.
"""

from pathlib import Path

import polars as pl

RAW = Path("data/processed/fraud_features.csv")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
TRAIN_N_FRAUD = 10_000
TRAIN_N_OK = 90_000
TEST_N_OK = 300_000
REAL_RATE = 0.00122
TEST_N_FRAUD = round(TEST_N_OK * REAL_RATE)  # tasa real de fraude


def main() -> None:
    lf = pl.scan_csv(RAW)

    # entrada limpia para el notebook: recomputamos log del monto (con signo,
    # porque hay abonos negativos) y el ratio monto/límite (con límite 0 en
    # tarjetas prepago); descartamos filas sin ventana temporal (primera
    # transacción de cada tarjeta).
    lf = lf.with_columns(
        (pl.col("amount").sign() * pl.col("amount").abs().log1p()).alias("amount_log"),
        pl.when(pl.col("credit_limit") == 0)
        .then(0.0)
        .otherwise(pl.col("amount") / pl.col("credit_limit"))
        .alias("ratio_amount_credit_limit"),
    ).filter(pl.col("time_since_last_txn_seconds").is_not_null())

    # particion determinista por id
    fraud = lf.filter(pl.col("is_fraud") == 1)
    ok = lf.filter(pl.col("is_fraud") == 0)

    # train: fraude sobremuestreado
    train_fraud = fraud.sort("id").collect().sample(n=TRAIN_N_FRAUD, seed=SEED, shuffle=True)
    train_ok = ok.sort("id").collect().sample(n=TRAIN_N_OK, seed=SEED, shuffle=True)
    train = pl.concat([train_fraud, train_ok]).sample(fraction=1.0, shuffle=True, seed=SEED)

    # test: tasa real de fraude (0.122 %), tomando el resto de no-fraude
    test_ok = (
        ok.sort("id")
        .join(pl.LazyFrame(train_ok.select("id")), on="id", how="anti")
        .collect()
        .sample(n=TEST_N_OK, seed=SEED, shuffle=True)
    )
    test_fraud = (
        fraud.sort("id")
        .join(pl.LazyFrame(train_fraud.select("id")), on="id", how="anti")
        .collect()
        .sample(n=TEST_N_FRAUD, seed=SEED, shuffle=True)
    )
    test = pl.concat([test_ok, test_fraud]).sample(fraction=1.0, shuffle=True, seed=SEED)

    train.write_csv(OUT / "fraud_train.csv")
    test.write_csv(OUT / "fraud_test.csv")

    for name, df in [("train", train), ("test", test)]:
        n = df.height
        f = df.filter(pl.col("is_fraud") == 1).height
        print(f"{name}: {n:,} filas | fraude {f:,} | tasa {f / n:.4%}")


if __name__ == "__main__":
    main()
