import pandas as pd
import numpy as np

np.random.seed(42)

# =========================================================
# 1. PRODUCT MASTER
# =========================================================

categories = [
    "Electronics",
    "Home & Kitchen",
    "Beauty",
    "Fashion",
    "Sports"
]

price_ranges = {
    "Electronics": (200, 1500),
    "Home & Kitchen": (50, 500),
    "Beauty": (20, 200),
    "Fashion": (30, 400),
    "Sports": (40, 600)
}

demand_ranges = {
    "Electronics": (10, 40),
    "Home & Kitchen": (20, 60),
    "Beauty": (40, 100),
    "Fashion": (30, 80),
    "Sports": (20, 60)
}

products = []

for i in range(50):
    category = categories[i // 10]

    products.append({
        "product_id": f"P{i+1:03d}",
        "category": category,
        "base_price": np.random.randint(*price_ranges[category]),
        "base_demand": np.random.randint(*demand_ranges[category])
    })

product_master = pd.DataFrame(products)


# =========================================================
# 2. GENERATE NORMAL DAILY DATA
# =========================================================

dates = pd.date_range(
    start="2025-01-01",
    periods=365
)

daily_rows = []

for _, product in product_master.iterrows():

    for date in dates:

        # Weekend effect
        is_weekend = date.dayofweek >= 5
        weekend_factor = 1.20 if is_weekend else 1.0

        # Random promotion
        promotion = np.random.rand() < 0.08

        if promotion:
            promotion_factor = 1.35
            discount_pct = np.random.uniform(0.05, 0.20)
        else:
            promotion_factor = 1.0
            discount_pct = 0

        # Daily price
        price = product["base_price"] * (1 - discount_pct)

        # Natural randomness
        noise_factor = np.random.normal(
            loc=1.0,
            scale=0.10
        )

        # Normal expected demand
        expected_demand = (
            product["base_demand"]
            * weekend_factor
            * promotion_factor
            * noise_factor
        )

        units_sold = max(
            0,
            int(round(expected_demand))
        )

        # Simple daily inventory level
        inventory = np.random.randint(
            low=max(units_sold, 1),
            high=max(units_sold * 5, 2)
        )

        revenue = units_sold * price

        daily_rows.append({
            "date": date,
            "product_id": product["product_id"],
            "category": product["category"],
            "base_price": product["base_price"],
            "base_demand": product["base_demand"],
            "price": round(price, 2),
            "promotion": int(promotion),
            "inventory": inventory,
            "units_sold": units_sold,
            "revenue": round(revenue, 2)
        })

df = pd.DataFrame(daily_rows)


# =========================================================
# 3. GROUND-TRUTH COLUMNS
# =========================================================

df["is_anomaly"] = 0
df["anomaly_type"] = "normal"


# =========================================================
# 4. ANOMALY EVENT GENERATOR
# =========================================================

rng = np.random.default_rng(42)

used_rows = set()


def get_event_indices(product_id, start_date, duration):
    mask = (
        (df["product_id"] == product_id)
        & (df["date"] >= start_date)
        & (df["date"] < start_date + pd.Timedelta(days=duration))
    )

    return df.index[mask].tolist()


def create_event(duration_range):
    """
    Pick a random product and a random date.
    Avoid overlap with previously injected anomalies.
    """

    while True:

        product_id = rng.choice(product_master["product_id"])

        duration = rng.integers(
            duration_range[0],
            duration_range[1] + 1
        )

        latest_start = len(dates) - duration

        start_pos = rng.integers(
            0,
            latest_start
        )

        start_date = dates[start_pos]

        indices = get_event_indices(
            product_id,
            start_date,
            duration
        )

        if not any(i in used_rows for i in indices):

            used_rows.update(indices)

            return indices


# =========================================================
# 5. STOCK-OUT EVENTS
# =========================================================

n_stockout_events = 10

for _ in range(n_stockout_events):

    indices = create_event((2, 5))

    df.loc[indices, "inventory"] = 0
    df.loc[indices, "units_sold"] = 0
    df.loc[indices, "revenue"] = 0

    df.loc[indices, "is_anomaly"] = 1
    df.loc[indices, "anomaly_type"] = "stockout"


# =========================================================
# 6. DEMAND DROP EVENTS
# =========================================================

n_demand_drop_events = 18

for _ in range(n_demand_drop_events):

    indices = create_event((1, 3))

    new_units = (
        df.loc[indices, "units_sold"]
        * rng.uniform(0.15, 0.40)
    ).round().astype(int)

    df.loc[indices, "units_sold"] = new_units.to_numpy()

    new_revenue = (
        df.loc[indices, "units_sold"].to_numpy()
        * df.loc[indices, "price"].to_numpy()
    )

    df.loc[indices, "revenue"] = np.round(
        new_revenue,
        2
    )

    df.loc[indices, "is_anomaly"] = 1
    df.loc[indices, "anomaly_type"] = "demand_drop"


# =========================================================
# 7. DEMAND SPIKE EVENTS
# =========================================================

n_demand_spike_events = 14

for _ in range(n_demand_spike_events):

    indices = create_event((1, 2))

    multiplier = rng.uniform(2.2, 3.5)

    new_units = (
        df.loc[indices, "units_sold"]
        * multiplier
    ).round().astype(int)

    df.loc[indices, "units_sold"] = new_units.to_numpy()

    new_revenue = (
        df.loc[indices, "units_sold"].to_numpy()
        * df.loc[indices, "price"].to_numpy()
    )

    df.loc[indices, "revenue"] = np.round(
        new_revenue,
        2
    )

    df.loc[indices, "is_anomaly"] = 1
    df.loc[indices, "anomaly_type"] = "demand_spike"


# =========================================================
# 8. PRICE INCREASE EVENTS
# =========================================================

n_price_events = 8

for _ in range(n_price_events):

    indices = create_event((2, 6))

    multiplier = rng.uniform(1.4, 1.9)

    new_price = (
        df.loc[indices, "price"]
        * multiplier
    ).round(2)

    df.loc[indices, "price"] = new_price.to_numpy()

    new_revenue = (
        df.loc[indices, "units_sold"].to_numpy()
        * df.loc[indices, "price"].to_numpy()
    )

    df.loc[indices, "revenue"] = np.round(
        new_revenue,
        2
    )

    df.loc[indices, "is_anomaly"] = 1
    df.loc[indices, "anomaly_type"] = "price_increase"


# =========================================================
# 9. PROMOTION UNDERPERFORMANCE EVENTS
# =========================================================

n_promo_events = 11

for _ in range(n_promo_events):

    indices = create_event((2, 4))

    df.loc[indices, "promotion"] = 1

    underperformance_factor = rng.uniform(
        0.40,
        0.70
    )

    new_units = (
        df.loc[indices, "base_demand"]
        * underperformance_factor
    ).round().astype(int)

    df.loc[indices, "units_sold"] = new_units.to_numpy()

    new_revenue = (
        df.loc[indices, "units_sold"].to_numpy()
        * df.loc[indices, "price"].to_numpy()
    )

    df.loc[indices, "revenue"] = np.round(
        new_revenue,
        2
    )

    df.loc[indices, "is_anomaly"] = 1
    df.loc[indices, "anomaly_type"] = "promo_underperformance"


# =========================================================
# 10. SAVE DATA
# =========================================================

df.to_csv(
    "ecommerce_data.csv",
    index=False
)


# =========================================================
# 11. BASIC CHECKS
# =========================================================

print(df.head())

print("\nShape:")
print(df.shape)

print("\nMissing values:")
print(df.isna().sum())

print("\nAnomaly counts:")
print(df["anomaly_type"].value_counts())

print("\nTotal anomalies:")
print(df["is_anomaly"].sum())

print("\nSummary:")
print(
    df[
        [
            "price",
            "inventory",
            "units_sold",
            "revenue"
        ]
    ].describe()
)