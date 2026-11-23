import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==================================================
# 1. LOAD ML-READY DATA
# ==================================================

df = pd.read_csv("ecommerce_features.csv")

# Ensure chronological order
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

features = [
    "demand_ratio",
    "price_ratio",
    "inventory_ratio",
    "promotion",
    "is_weekend",
    "revenue_ratio"
]

X = df[features]
y_true = df["is_anomaly"]


# ==================================================
# 2. HISTORICAL STATISTICAL BASELINE
# ==================================================

# Continuous features used for Z-score analysis
stat_features = [
    "demand_ratio",
    "price_ratio",
    "inventory_ratio",
    "revenue_ratio"
]

# --------------------------------------------------
# Historical Rolling Z-Scores
# --------------------------------------------------

for feature in stat_features:

    # Historical mean:
    # same product + same weekday/weekend context
    # using previous observations only
    historical_mean = (
        df.groupby(["product_id", "is_weekend"])[feature]
        .transform(
            lambda x: x.shift(1)
                       .rolling(window=30, min_periods=5)
                       .mean()
        )
    )

    # Historical standard deviation
    historical_std = (
        df.groupby(["product_id", "is_weekend"])[feature]
        .transform(
            lambda x: x.shift(1)
                       .rolling(window=30, min_periods=5)
                       .std()
        )
    )

    # Today's deviation from historical behavior
    df[f"{feature}_zscore"] = (
        (df[feature] - historical_mean)
        / historical_std
    )


# --------------------------------------------------
# Statistical Anomaly Score
# --------------------------------------------------

zscore_columns = [
    "demand_ratio_zscore",
    "price_ratio_zscore",
    "inventory_ratio_zscore",
    "revenue_ratio_zscore"
]

# Strongest absolute KPI deviation
df["stat_score"] = (
    df[zscore_columns]
    .abs()
    .max(axis=1)
)


# --------------------------------------------------
# Final Statistical Baseline
# --------------------------------------------------

stat_threshold = 4.5

df["stat_anomaly"] = (
    df["stat_score"] > stat_threshold
).astype(int)

stat_pred = df["stat_anomaly"]

stat_precision = precision_score(y_true, stat_pred)
stat_recall = recall_score(y_true, stat_pred)
stat_f1 = f1_score(y_true, stat_pred)

print("\n--- Statistical Baseline Results ---")

print("Predicted anomalies:", stat_pred.sum())
print("Precision:", round(stat_precision, 3))
print("Recall:", round(stat_recall, 3))
print("F1:", round(stat_f1, 3))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, stat_pred))


# --------------------------------------------------
# Statistical Detection by Type
# --------------------------------------------------

print("\n--- Statistical Detection by Type ---")

stat_type_results = (
    df[df["is_anomaly"] == 1]
    .groupby("anomaly_type")["stat_anomaly"]
    .agg(["count", "sum"])
)

stat_type_results["recall"] = (
    stat_type_results["sum"]
    / stat_type_results["count"]
)

stat_type_results.columns = [
    "actual_anomalies",
    "detected",
    "recall"
]

print(stat_type_results.round(3))


# ==================================================
# 3. ISOLATION FOREST
# ==================================================

if_model = IsolationForest(
    n_estimators=100,
    max_samples=4096,
    max_features=1.0,
    contamination=0.01,
    random_state=42
)

df["if_prediction"] = if_model.fit_predict(X)

# Isolation Forest:
#  1 = normal
# -1 = anomaly
df["if_anomaly"] = (
    df["if_prediction"] == -1
).astype(int)

# Lower score = more anomalous
df["if_score"] = if_model.decision_function(X)


# --------------------------------------------------
# Isolation Forest Evaluation
# --------------------------------------------------

if_pred = df["if_anomaly"]

if_precision = precision_score(y_true, if_pred)
if_recall = recall_score(y_true, if_pred)
if_f1 = f1_score(y_true, if_pred)

print("\n--- Isolation Forest Results ---")

print("Predicted anomalies:", if_pred.sum())
print("Precision:", round(if_precision, 3))
print("Recall:", round(if_recall, 3))
print("F1:", round(if_f1, 3))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, if_pred))


# --------------------------------------------------
# Isolation Forest Detection by Type
# --------------------------------------------------

print("\n--- Isolation Forest Detection by Type ---")

if_type_results = (
    df[df["is_anomaly"] == 1]
    .groupby("anomaly_type")["if_anomaly"]
    .agg(["count", "sum"])
)

if_type_results["recall"] = (
    if_type_results["sum"]
    / if_type_results["count"]
)

if_type_results.columns = [
    "actual_anomalies",
    "detected",
    "recall"
]

print(if_type_results.round(3))


# ==================================================
# 4. LOCAL OUTLIER FACTOR
# ==================================================

# --------------------------------------------------
# Standardize Features
# --------------------------------------------------

# LOF is distance-based, so features should be
# on comparable scales.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# --------------------------------------------------
# Tuned LOF
# --------------------------------------------------

lof_model = LocalOutlierFactor(
    n_neighbors=1000,
    contamination=0.01
)

df["lof_prediction"] = lof_model.fit_predict(X_scaled)

# LOF:
#  1 = normal
# -1 = anomaly
df["lof_anomaly"] = (
    df["lof_prediction"] == -1
).astype(int)


# --------------------------------------------------
# LOF Evaluation
# --------------------------------------------------

lof_pred = df["lof_anomaly"]

lof_precision = precision_score(y_true, lof_pred)
lof_recall = recall_score(y_true, lof_pred)
lof_f1 = f1_score(y_true, lof_pred)

print("\n--- LOF Results ---")

print("Predicted anomalies:", lof_pred.sum())
print("Precision:", round(lof_precision, 3))
print("Recall:", round(lof_recall, 3))
print("F1:", round(lof_f1, 3))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, lof_pred))


# --------------------------------------------------
# LOF Detection by Type
# --------------------------------------------------

print("\n--- LOF Detection by Type ---")

lof_type_results = (
    df[df["is_anomaly"] == 1]
    .groupby("anomaly_type")["lof_anomaly"]
    .agg(["count", "sum"])
)

lof_type_results["recall"] = (
    lof_type_results["sum"]
    / lof_type_results["count"]
)

lof_type_results.columns = [
    "actual_anomalies",
    "detected",
    "recall"
]

print(lof_type_results.round(3))


# ==================================================
# 5. FINAL MODEL COMPARISON
# ==================================================

comparison = pd.DataFrame({
    "Model": [
        "Statistical Baseline",
        "Isolation Forest",
        "LOF"
    ],
    "Precision": [
        stat_precision,
        if_precision,
        lof_precision
    ],
    "Recall": [
        stat_recall,
        if_recall,
        lof_recall
    ],
    "F1": [
        stat_f1,
        if_f1,
        lof_f1
    ]
})

print("\n--- Final Model Comparison ---")
print(
    comparison
    .round(3)
    .to_string(index=False)
)



# ==================================================
# 5. EXPORT LOF-DETECTED ANOMALIES
# ==================================================

# LOF scores: lower values indicate more abnormal observations
df["lof_score"] = lof_model.negative_outlier_factor_

# Keep only observations detected as anomalies
detected_anomalies = df[df["lof_anomaly"] == 1].copy()

# Rank most abnormal observations first
detected_anomalies = detected_anomalies.sort_values(
    "lof_score",
    ascending=True
)

# Export only the information needed for investigation
detected_anomalies[
    ["product_id", "date", "lof_score"]
].to_csv(
    "lof_detected_anomalies.csv",
    index=False
)

print("\n--- LOF Export ---")
print("Exported anomalies:", len(detected_anomalies))
print(
    detected_anomalies[
        ["product_id", "date", "lof_score"]
    ].head()
)