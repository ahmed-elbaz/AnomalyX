import pandas as pd


# ==================================================
# 1. LOAD DATA
# ==================================================

df = pd.read_csv("ecommerce_features.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["product_id", "date"]).reset_index(drop=True)


# ==================================================
# 2. TOOL: GET PRODUCT CONTEXT
# ==================================================

def get_product_context(product_id, date):

    date = pd.to_datetime(date)

    row = df[
        (df["product_id"] == product_id)
        & (df["date"] == date)
    ]

    if row.empty:
        return {
            "status": "not_found",
            "message": f"No data found for {product_id} on {date.date()}"
        }

    row = row.iloc[0]

    return {
        "product_id": row["product_id"],
        "date": str(row["date"].date()),
        "category": row["category"],

        "units_sold": int(row["units_sold"]),
        "demand_ratio": round(float(row["demand_ratio"]), 3),

        "price": round(float(row["price"]), 2),
        "price_ratio": round(float(row["price_ratio"]), 3),

        "inventory": int(row["inventory"]),
        "inventory_ratio": round(float(row["inventory_ratio"]), 3),

        "promotion": int(row["promotion"]),

        "revenue": round(float(row["revenue"]), 2),
        "revenue_ratio": round(float(row["revenue_ratio"]), 3)
    }


# ==================================================
# 3. TOOL: GET HISTORICAL CONTEXT
# ==================================================

def get_historical_context(product_id, date, window_days=30):

    date = pd.to_datetime(date)

    history = df[
        (df["product_id"] == product_id)
        & (df["date"] < date)
        & (df["date"] >= date - pd.Timedelta(days=window_days))
    ]

    if history.empty:
        return {
            "status": "not_found",
            "message": f"No historical data found for {product_id}"
        }

    return {
        "product_id": product_id,
        "date": str(date.date()),
        "history_days": len(history),

        "avg_units_sold": round(
            float(history["units_sold"].mean()), 2
        ),

        "avg_price": round(
            float(history["price"].mean()), 2
        ),

        "avg_inventory": round(
            float(history["inventory"].mean()), 2
        ),

        "avg_revenue": round(
            float(history["revenue"].mean()), 2
        ),

        "promotion_days": int(
            history["promotion"].sum()
        )
    }


# ==================================================
# 4. TOOL: COMPARE WITH PEERS
# ==================================================

def compare_with_peers(product_id, date):

    date = pd.to_datetime(date)

    target = df[
        (df["product_id"] == product_id)
        & (df["date"] == date)
    ]

    if target.empty:
        return {
            "status": "not_found",
            "message": f"No data found for {product_id} on {date.date()}"
        }

    target = target.iloc[0]
    category = target["category"]

    # Same-category products on the same date,
    # excluding the target product.
    peers = df[
        (df["category"] == category)
        & (df["date"] == date)
        & (df["product_id"] != product_id)
    ]

    if peers.empty:
        return {
            "status": "not_found",
            "message": f"No peers found for {product_id} on {date.date()}"
        }

    return {
        "product_id": product_id,
        "date": str(date.date()),
        "category": category,
        "peer_count": len(peers),

        "product_demand_ratio": round(
            float(target["demand_ratio"]), 3
        ),

        "peer_avg_demand_ratio": round(
            float(peers["demand_ratio"].mean()), 3
        ),

        "product_revenue_ratio": round(
            float(target["revenue_ratio"]), 3
        ),

        "peer_avg_revenue_ratio": round(
            float(peers["revenue_ratio"].mean()), 3
        )
    }


# ==================================================
# 5. TOOL: GET PROMOTION CONTEXT
# ==================================================

def get_promotion_context(product_id, date, window_days=90):

    date = pd.to_datetime(date)

    target = df[
        (df["product_id"] == product_id)
        & (df["date"] == date)
    ]

    if target.empty:
        return {
            "status": "not_found",
            "message": f"No data found for {product_id} on {date.date()}"
        }

    target = target.iloc[0]

    # Previous history only.
    history = df[
        (df["product_id"] == product_id)
        & (df["date"] < date)
        & (df["date"] >= date - pd.Timedelta(days=window_days))
    ]

    # Keep only previous promotional days.
    promo_history = history[
        history["promotion"] == 1
    ]

    if promo_history.empty:
        return {
            "product_id": product_id,
            "date": str(date.date()),
            "current_promotion": int(target["promotion"]),
            "previous_promo_days": 0,
            "message": "No previous promotion days available for comparison"
        }

    return {
        "product_id": product_id,
        "date": str(date.date()),

        "current_promotion": int(target["promotion"]),
        "current_units_sold": int(target["units_sold"]),
        "current_demand_ratio": round(
            float(target["demand_ratio"]), 3
        ),

        "previous_promo_days": len(promo_history),

        "previous_promo_avg_units_sold": round(
            float(promo_history["units_sold"].mean()), 2
        ),

        "previous_promo_avg_demand_ratio": round(
            float(promo_history["demand_ratio"].mean()), 3
        )
    }