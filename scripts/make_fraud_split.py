"""Genera las 4 particiones del dataset de fraude (ids disjuntos).

Salidas (gitignored por `data/processed/fraud*`):
- fraud_train.csv     ~250k,  ~10 % fraude (búsqueda GA).
- fraud_train_big.csv ~2 M,   1.3 % fraude (reentrenamiento final).
- fraud_val.csv       ~1.1 M, tasa real (umbral + fitness).
- fraud_test.csv      ~2 M,   tasa real (held-out, FPR final).

La partición de fraude es exacta (2400 test, 1200 val, resto train). La de
no-fraude usa buckets por hash del id: determinista y disjunta.
"""

from pathlib import Path

import polars as pl

RAW = Path("data/processed/fraud_features.csv")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_TEST_FRAUD = 2400
N_VAL_FRAUD = 1200

# buckets no-fraude (hash % 200): [lo, hi)
TEST_BUCKET = (0, 17)
VAL_BUCKET = (17, 26)
BIG_BUCKET = (26, 43)
TRAIN_BUCKET = (43, 45)


def main() -> None:
    lf = pl.scan_csv(RAW)

    # entrada limpia (mismo saneo que make_fraud_sample.py)
    lf = lf.with_columns(
        (pl.col("amount").sign() * pl.col("amount").abs().log1p()).alias("amount_log"),
        pl.when(pl.col("credit_limit") == 0)
        .then(0.0)
        .otherwise(pl.col("amount") / pl.col("credit_limit"))
        .alias("ratio_amount_credit_limit"),
    ).filter(pl.col("time_since_last_txn_seconds").is_not_null())

    COLS = list(lf.collect_schema().names())

    # ---- fraude: particion exacta ----
    fraud = lf.filter(pl.col("is_fraud") == 1).collect().sample(
        fraction=1.0, shuffle=True, seed=SEED
    )
    test_fraud = fraud[:N_TEST_FRAUD]
    val_fraud = fraud[N_TEST_FRAUD:N_TEST_FRAUD + N_VAL_FRAUD]
    train_fraud = fraud[N_TEST_FRAUD + N_VAL_FRAUD:]

    # ---- no-fraude: buckets hash disjuntos ----
    ok = lf.filter(pl.col("is_fraud") == 0).with_columns(
        (pl.col("id").hash(seed=SEED) % 200).alias("b")
    )

    def ok_part(lo: int, hi: int) -> pl.LazyFrame:
        return ok.filter((pl.col("b") >= lo) & (pl.col("b") < hi)).select(COLS)

    def write(name: str, fraud_df: pl.DataFrame, ok_partition: pl.LazyFrame) -> None:
        full = pl.concat([pl.LazyFrame(fraud_df), ok_partition])
        full.sink_csv(OUT / name)

    write("fraud_test.csv", test_fraud, ok_part(*TEST_BUCKET))
    write("fraud_val.csv", val_fraud, ok_part(*VAL_BUCKET))
    write("fraud_train_big.csv", train_fraud, ok_part(*BIG_BUCKET))
    write("fraud_train.csv", train_fraud, ok_part(*TRAIN_BUCKET))

    for name in ["fraud_train.csv", "fraud_train_big.csv",
                 "fraud_val.csv", "fraud_test.csv"]:
        c = pl.scan_csv(OUT / name)
        n = c.select(pl.len()).collect().item()
        f = c.filter(pl.col("is_fraud") == 1).select(pl.len()).collect().item()
        print(f"{name}: {n:,} filas | fraude {f:,} | tasa {f / n:.4%}")


if __name__ == "__main__":
    main()
