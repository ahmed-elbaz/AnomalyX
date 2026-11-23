import pandas as pd

# Load data
df = pd.read_csv("ecommerce_data.csv")
df["date"] = pd.to_datetime(df["date"])

# Important for historical calculations
df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

# =========================================================
# Historical baselines
# Previous 30 days, NOT including today
# =========================================================

df["sales_baseline_30d"] = (
    df.groupby("product_id")["units_sold"]
      .transform(lambda x: x.shift(1).rolling(30, min_periods=7).mean())
)

df["price_baseline_30d"] = (
    df.groupby("product_id")["price"]
      .transform(lambda x: x.shift(1).rolling(30, min_periods=7).mean())
)

df["inventory_baseline_30d"] = (
    df.groupby("product_id")["inventory"]
      .transform(lambda x: x.shift(1).rolling(30, min_periods=7).mean())
)

df["revenue_baseline_30d"] = (
    df.groupby("product_id")["revenue"]
      .transform(lambda x: x.shift(1).rolling(30, min_periods=7).mean())
)

# =========================================================
# Relative features
# =========================================================

df["demand_ratio"] = (
    df["units_sold"] / df["sales_baseline_30d"]
)

df["price_ratio"] = (
    df["price"] / df["price_baseline_30d"]
)

df["inventory_ratio"] = (
    df["inventory"] / df["inventory_baseline_30d"]
)

df["revenue_ratio"] = (
    df["revenue"] / df["revenue_baseline_30d"]
)

# Context
df["is_weekend"] = (
    df["date"].dt.dayofweek >= 5
).astype(int)

# =========================================================
# Features ML will eventually see
# =========================================================

feature_columns = [
    "demand_ratio",
    "price_ratio",
    "inventory_ratio",
    "promotion",
    "is_weekend",
    "revenue_ratio"
]

# Remove early rows without enough history
ml_df = df.dropna(subset=feature_columns).copy()

print("\nOriginal rows:", len(df))
print("ML-ready rows:", len(ml_df))

print("\nNormal examples:")
print(
    ml_df[ml_df["is_anomaly"] == 0][
        ["date", "product_id", "units_sold",
         "sales_baseline_30d", "demand_ratio"]
    ].head(10)
)

print("\nAnomaly examples:")
print(
    ml_df[ml_df["is_anomaly"] == 1][
        ["date", "product_id", "anomaly_type",
         "units_sold", "sales_baseline_30d",
         "demand_ratio"]
    ].head(15)
)

print("\nAverage features by anomaly type:")

print(
    ml_df.groupby("anomaly_type")[
        [
            "demand_ratio",
            "price_ratio",
            "inventory_ratio",
            "promotion",
            "is_weekend",
            "revenue_ratio"
        ]
    ].mean().round(2)
)

ml_df.to_csv("ecommerce_features.csv", index=False)