"""Genera el dataset de features de fraude (IBM SD254 v2).

Salida: data/processed/fraud_features.csv con columnas
[id, features numericas..., is_fraud].

- Procesamiento con polars (lazy + streaming).
- Solo codificaciones puras: one-hot, ordinal, banderas binarias.
- Sin imputacion: los nulos derivados quedan como nulos.
"""

from pathlib import Path

import polars as pl

RAW = Path("data/raw/ibm")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)


def money(col: str) -> pl.Expr:
    return pl.col(col).str.replace_all(r"[$,]", "").cast(pl.Float64)


def main() -> None:
    # ---------------- dimensiones (eager, pequenas) ----------------
    cards = pl.read_csv(RAW / "sd254_cards.csv")
    users = pl.read_csv(RAW / "sd254_users.csv").with_row_index("User")

    cards = cards.with_columns(
        [
            money("Credit Limit").alias("credit_limit"),
            pl.col("Acct Open Date").str.slice(0, 2).cast(pl.Int16).alias("open_month"),
            pl.col("Acct Open Date").str.slice(4, 4).cast(pl.Int16).alias("open_year"),
        ]
    ).select(
        [
            "User",
            "CARD INDEX",
            "Card Brand",
            "Card Type",
            "Has Chip",
            "credit_limit",
            "open_month",
            "open_year",
        ]
    )

    users = users.select(
        [
            "User",
            "FICO Score",
            "Total Debt",
            "Yearly Income - Person",
            "Per Capita Income - Zipcode",
            "Current Age",
            "Num Credit Cards",
            "State",
            "Zipcode",
        ]
    ).rename(
        {
            "FICO Score": "fico_score",
            "Total Debt": "total_debt",
            "Yearly Income - Person": "yearly_income",
            "Per Capita Income - Zipcode": "per_capita_income_zip",
            "Current Age": "current_age",
            "Num Credit Cards": "num_credit_cards",
            "State": "user_state",
            "Zipcode": "user_zipcode",
        }
    )

    # ---------------- transacciones (lazy, streaming) ----------------
    tx = pl.scan_csv(RAW / "credit_card_transactions-ibm_v2.csv").with_row_index("id")

    # parseo de campos crudos
    tx = tx.with_columns(
        [
            money("Amount").alias("amount"),
            pl.col("Time").str.slice(0, 2).cast(pl.Int8).alias("hour"),
            pl.col("Time").str.slice(3, 2).cast(pl.Int8).alias("minute"),
        ]
    )

    # ordinal de MCC
    mcc_map = (
        tx.select("MCC")
        .unique()
        .sort("MCC")
        .with_row_index("mcc_ord")
        .collect()
    )
    tx = tx.join(pl.LazyFrame(mcc_map), on="MCC", how="left")

    # join dimensiones
    tx = tx.join(
        pl.LazyFrame(cards),
        left_on=["User", "Card"],
        right_on=["User", "CARD INDEX"],
        how="left",
    )
    tx = tx.join(pl.LazyFrame(users), on="User", how="left")

    # timestamp para ordenar por tarjeta
    tx = tx.with_columns(
        pl.datetime("Year", "Month", "Day", "hour", "minute").dt.epoch("s").alias("ts")
    )

    # ---------------- block 1: transaccion ----------------
    err = pl.col("Errors?")
    tx = tx.with_columns(
        [
            (pl.col("amount").sign() * pl.col("amount").abs().log1p()).alias("amount_log"),
            # canal
            (pl.col("Use Chip") == "Online Transaction").cast(pl.Int8).alias("use_chip_online"),
            (pl.col("Use Chip") == "Chip Transaction").cast(pl.Int8).alias("use_chip_chip"),
            (pl.col("Use Chip") == "Swipe Transaction").cast(pl.Int8).alias("use_chip_swipe"),
            # errores (banderas binarias)
            err.str.contains("Bad CVV").fill_null(False).cast(pl.Int8).alias("err_bad_cvv"),
            err.str.contains("Bad PIN").fill_null(False).cast(pl.Int8).alias("err_bad_pin"),
            err.str.contains("Bad Card Number").fill_null(False).cast(pl.Int8).alias("err_bad_card_number"),
            err.str.contains("Bad Expiration").fill_null(False).cast(pl.Int8).alias("err_bad_expiration"),
            err.str.contains("Bad Zipcode").fill_null(False).cast(pl.Int8).alias("err_bad_zipcode"),
            err.str.contains("Insufficient Balance").fill_null(False).cast(pl.Int8).alias("err_insufficient_balance"),
            err.str.contains("Technical Glitch").fill_null(False).cast(pl.Int8).alias("err_technical_glitch"),
            err.is_null().cast(pl.Int8).alias("err_none"),
            # tiempo
            pl.date("Year", "Month", "Day").dt.weekday().cast(pl.Int8).alias("day_of_week"),
            pl.col("Month").cast(pl.Int8).alias("month"),
        ]
    )

    # ---------------- block 2: tarjeta ----------------
    tx = tx.with_columns(
        [
            (pl.col("Card Type") == "Debit").cast(pl.Int8).alias("card_type_debit"),
            (pl.col("Card Type") == "Credit").cast(pl.Int8).alias("card_type_credit"),
            (pl.col("Card Type") == "Debit (Prepaid)").cast(pl.Int8).alias("card_type_debit_prepaid"),
            (pl.col("Card Brand") == "Visa").cast(pl.Int8).alias("card_brand_visa"),
            (pl.col("Card Brand") == "Mastercard").cast(pl.Int8).alias("card_brand_mastercard"),
            (pl.col("Card Brand") == "Amex").cast(pl.Int8).alias("card_brand_amex"),
            (pl.col("Card Brand") == "Discover").cast(pl.Int8).alias("card_brand_discover"),
            (pl.col("Has Chip") == "YES").cast(pl.Int8).alias("has_chip"),
            pl.when(pl.col("credit_limit") == 0)
            .then(0.0)
            .otherwise(pl.col("amount") / pl.col("credit_limit"))
            .alias("ratio_amount_credit_limit"),
            (
                (pl.col("Year") * 12 + pl.col("Month"))
                - (pl.col("open_year") * 12 + pl.col("open_month"))
            ).cast(pl.Int16).alias("card_age_months"),
        ]
    )

    # ---------------- block 3: titular ----------------
    tx = tx.with_columns(
        [
            money("total_debt").alias("total_debt"),
            money("yearly_income").alias("yearly_income"),
            money("per_capita_income_zip").alias("per_capita_income_zip"),
        ]
    )
    tx = tx.with_columns(
        (pl.col("total_debt") / pl.col("yearly_income")).alias("debt_to_income_ratio")
    )

    # ---------------- block 4: comportamiento ----------------
    # z-score del monto por usuario
    user_amt = (
        tx.select(["User", "amount"])
        .group_by("User")
        .agg(
            pl.col("amount").mean().alias("user_amt_mean"),
            pl.col("amount").std().alias("user_amt_std"),
        )
        .collect()
    )
    tx = tx.join(pl.LazyFrame(user_amt), on="User", how="left")
    tx = tx.with_columns(
        ((pl.col("amount") - pl.col("user_amt_mean")) / pl.col("user_amt_std")).alias("amount_z_user")
    )

    # velocidad por tarjeta (dia y hora)
    card_day = (
        tx.select(["User", "Card", "Year", "Month", "Day"])
        .group_by(["User", "Card", "Year", "Month", "Day"])
        .len()
        .rename({"len": "card_txn_count_day"})
    )
    card_hour = (
        tx.select(["User", "Card", "Year", "Month", "Day", "hour"])
        .group_by(["User", "Card", "Year", "Month", "Day", "hour"])
        .len()
        .rename({"len": "card_txn_count_hour"})
    )
    tx = tx.join(card_day, on=["User", "Card", "Year", "Month", "Day"], how="left")
    tx = tx.join(card_hour, on=["User", "Card", "Year", "Month", "Day", "hour"], how="left")

    # frecuencia del MCC por usuario
    user_mcc = (
        tx.select(["User", "MCC"])
        .group_by(["User", "MCC"])
        .len()
        .rename({"len": "user_mcc_count"})
    )
    tx = tx.join(user_mcc, on=["User", "MCC"], how="left")
    tx = tx.with_columns(
        (pl.col("user_mcc_count") == 1).cast(pl.Int8).alias("mcc_is_new_user")
    )

    # desajuste geografico (zip del comercio vs zip del titular)
    tx = tx.with_columns(
        [
            (
                pl.col("Merchant State").is_not_null()
                & pl.col("user_state").is_not_null()
                & (pl.col("Merchant State") != pl.col("user_state"))
            )
            .cast(pl.Int8)
            .alias("merchant_state_mismatch"),
            (
                pl.col("Zip").is_not_null()
                & (pl.col("Zip").cast(pl.Int64) == pl.col("user_zipcode").cast(pl.Int64))
            )
            .cast(pl.Int8)
            .alias("merchant_zip_match"),
        ]
    )

    # tiempo desde la ultima transaccion (segundos) por tarjeta
    tsl = (
        tx.select(["id", "User", "Card", "ts"])
        .sort(["User", "Card", "ts"])
        .with_columns(pl.col("ts").diff().over(["User", "Card"]).alias("time_since_last_txn_seconds"))
        .select(["id", "time_since_last_txn_seconds"])
        .collect()
    )
    tx = tx.join(pl.LazyFrame(tsl), on="id", how="left")

    # target
    tx = tx.with_columns((pl.col("Is Fraud?") == "Yes").cast(pl.Int8).alias("is_fraud"))

    # ---------------- seleccion y tipos finales ----------------
    float_cols = [
        "amount",
        "amount_log",
        "credit_limit",
        "ratio_amount_credit_limit",
        "total_debt",
        "yearly_income",
        "debt_to_income_ratio",
        "per_capita_income_zip",
        "amount_z_user",
        "time_since_last_txn_seconds",
    ]
    int_cols = [
        "mcc_ord",
        "user_mcc_count",
        "card_txn_count_day",
        "card_txn_count_hour",
        "fico_score",
        "current_age",
        "num_credit_cards",
    ]

    feature_cols = [
        # block 1
        "amount", "amount_log",
        "use_chip_online", "use_chip_chip", "use_chip_swipe",
        "mcc_ord",
        "err_bad_cvv", "err_bad_pin", "err_bad_card_number", "err_bad_expiration",
        "err_bad_zipcode", "err_insufficient_balance", "err_technical_glitch", "err_none",
        "hour", "day_of_week", "month",
        # block 2
        "card_type_debit", "card_type_credit", "card_type_debit_prepaid",
        "card_brand_visa", "card_brand_mastercard", "card_brand_amex", "card_brand_discover",
        "has_chip", "credit_limit", "ratio_amount_credit_limit", "card_age_months",
        # block 3
        "fico_score", "total_debt", "yearly_income", "debt_to_income_ratio",
        "per_capita_income_zip", "current_age", "num_credit_cards",
        # block 4
        "amount_z_user", "card_txn_count_day", "card_txn_count_hour",
        "merchant_state_mismatch", "merchant_zip_match",
        "user_mcc_count", "mcc_is_new_user", "time_since_last_txn_seconds",
    ]

    dtype = {c: pl.Float32 for c in float_cols}
    dtype.update({c: pl.Int32 for c in int_cols})
    cast_exprs = [
        pl.col(c).cast(dtype[c]).alias(c) if c in dtype else pl.col(c)
        for c in feature_cols
    ]

    out = tx.select(["id"] + feature_cols + ["is_fraud"]).with_columns(cast_exprs)

    # ---------------- escritura (orden determinista por id) ----------------
    out.sort("id").sink_csv(OUT / "fraud_features.csv")

    # ---------------- verificacion ----------------
    schema = out.collect_schema()
    print("Columnas:", len(schema))
    for name, dtype in schema.items():
        print(f"  {name}: {dtype}")

    check = pl.scan_csv(OUT / "fraud_features.csv")
    n = check.select(pl.len()).collect().item()
    fraud = check.filter(pl.col("is_fraud") == 1).select(pl.len()).collect().item()
    print(f"\nFilas: {n:,}  |  Fraudes: {fraud:,}  |  tasa: {fraud / n:.4%}")


if __name__ == "__main__":
    main()
