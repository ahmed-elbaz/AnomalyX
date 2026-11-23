import re

def calculate_change(current, historical):
    if historical is None or historical == 0:
        return None

    return round((current / historical - 1) * 100, 1)


def build_evidence_summary(product, history):
    return {
        "product_id": product["product_id"],
        "date": product["date"],

        "units_change_pct": calculate_change(
            product["units_sold"],
            history["avg_units_sold"]
        ),

        "price_change_pct": calculate_change(
            product["price"],
            history["avg_price"]
        ),

        "inventory_change_pct": calculate_change(
            product["inventory"],
            history["avg_inventory"]
        ),

        "revenue_change_pct": calculate_change(
            product["revenue"],
            history["avg_revenue"]
        )
    }


if __name__ == "__main__":
    product = {
        "product_id": "P035",
        "date": "2025-12-16",
        "units_sold": 53,
        "price": 517.13,
        "inventory": 156,
        "revenue": 27407.89
    }

    history = {
        "avg_units_sold": 62.5,
        "avg_price": 295.59,
        "avg_inventory": 188,
        "avg_revenue": 18446.26
    }

    summary = build_evidence_summary(product, history)

    print(summary)




def describe_evidence(evidence):
    descriptions = []

    metrics = {
        "units_change_pct": "Units sold",
        "price_change_pct": "Price",
        "inventory_change_pct": "Inventory",
        "revenue_change_pct": "Revenue"
    }

    for key, label in metrics.items():
        value = evidence.get(key)

        if value is None:
            continue

        if value > 0:
            direction = "increased"
        elif value < 0:
            direction = "decreased"
        else:
            direction = "remained unchanged"

        descriptions.append(
            f"{label} {direction} by {abs(value)}% "
            "compared with the historical average."
        )

    return descriptions



def check_direction_contradictions(report, evidence):
    contradictions = []

    metrics = {
        "units_change_pct": "units sold",
        "price_change_pct": "price",
        "inventory_change_pct": "inventory",
        "revenue_change_pct": "revenue"
    }

    for key, label in metrics.items():
        value = evidence.get(key)

        if value is None or value == 0:
            continue

        # Find statements linking a metric, direction and percentage
        pattern = (
            rf"\b{re.escape(label)}\b"
            rf"[^\n.!?]{{0,90}}?"
            rf"\b(increase|increased|decrease|decreased|rise|rose|drop|dropped)\b"
            rf"[^\n.!?]{{0,35}}?"
            rf"(\d+(?:\.\d+)?)%"
        )

        for match in re.finditer(pattern, report, flags=re.IGNORECASE):
            direction = match.group(1).lower()
            reported_value = float(match.group(2))

            # Only compare percentages matching the verified magnitude
            if abs(reported_value - abs(value)) > 0.2:
                continue

            is_increase = direction in {"increase", "increased", "rise", "rose"}

            if (value > 0 and not is_increase) or (value < 0 and is_increase):
                contradictions.append(
                    f"{label}: verified {value:+.1f}%, "
                    f"but report says '{match.group(0)}'"
                )

    return contradictions

