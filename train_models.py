import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
    mean_absolute_error, mean_squared_error, r2_score
)

# ==========================================================
# 1) LOAD DATASET
# ==========================================================
df = pd.read_csv("data/sri_lanka_flood_risk_dataset_25000.csv").drop_duplicates()
print("Dataset shape:", df.shape)

# ==========================================================
# 2) DEFINE TARGETS
# ==========================================================
target_cls = "flood_occurrence_current_event"
target_reg = "inundation_area_sqm"

df[target_cls] = df[target_cls].map({"Yes": 1, "No": 0})
df[target_reg] = df[target_reg].fillna(0)

# ==========================================================
# 3) DROP ID + DECISION LABEL COLUMNS (avoid leakage)
# ==========================================================
drop_cols = ["record_id", "place_name", "generation_date",
             "is_good_to_live", "reason_not_good_to_live"]

df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# ==========================================================
# SAFE NORMALIZATION FUNCTION (avoids division-by-zero)
# ==========================================================
def normalize(col):
    denom = col.max() - col.min()
    return (col - col.min()) / denom if denom != 0 else 0

# ==========================================================
# PART A — FLOOD OCCURRENCE MODEL (CLASSIFICATION)
# ==========================================================
X_cls = df.drop(columns=[target_cls, target_reg], errors="ignore")
y_cls = df[target_cls]

cat_cols = X_cls.select_dtypes(include="object").columns
num_cols = X_cls.select_dtypes(include="number").columns

# Fill missing values
for col in cat_cols:
    X_cls[col] = X_cls[col].fillna(X_cls[col].mode()[0])

for col in num_cols:
    X_cls[col] = X_cls[col].fillna(X_cls[col].median())

# Preprocessing
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

# Train/Test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y_cls, test_size=0.2, stratify=y_cls, random_state=42
)

# Classification models
models_cls = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
}

trained_cls = {}
cls_results = []

for name, model in models_cls.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    trained_cls[name] = pipe

    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]

    cls_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, prob)
    })

cls_results_df = pd.DataFrame(cls_results).sort_values("F1", ascending=False)

print("\n=== CLASSIFICATION MODEL RESULTS ===")
print(cls_results_df)

# Best classifier
best_cls_name = cls_results_df.iloc[0]["Model"]
best_cls_model = trained_cls[best_cls_name]
print("\nBest classifier:", best_cls_name)

best_pred = best_cls_model.predict(X_test)
best_prob = best_cls_model.predict_proba(X_test)[:, 1]

# Confusion matrix plot
ConfusionMatrixDisplay(confusion_matrix(y_test, best_pred)).plot()
plt.title(f"Confusion Matrix ({best_cls_name})")
plt.show()

# ROC curve plot
RocCurveDisplay.from_predictions(y_test, best_prob)
plt.title(f"ROC Curve ({best_cls_name})")
plt.show()

# ==========================================================
# PART B — FLOOD IMPACT MODEL (INUNDATION REGRESSION)
# ==========================================================
flood_df = df[df[target_cls] == 1].copy()

X_reg = flood_df.drop(columns=[target_cls, target_reg])
y_reg = flood_df[target_reg]

# log transform for stability
y_reg_log = np.log1p(y_reg)

cat_cols_r = X_reg.select_dtypes(include="object").columns
num_cols_r = X_reg.select_dtypes(include="number").columns

for col in cat_cols_r:
    X_reg[col] = X_reg[col].fillna(X_reg[col].mode()[0])

for col in num_cols_r:
    X_reg[col] = X_reg[col].fillna(X_reg[col].median())

preprocessor_r = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_r),
    ("num", "passthrough", num_cols_r)
])

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg_log, test_size=0.2, random_state=42
)

# Regression models
models_reg = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}

trained_reg = {}
reg_results = []

for name, model in models_reg.items():
    pipe = Pipeline([("preprocessor", preprocessor_r), ("model", model)])
    pipe.fit(Xr_train, yr_train)
    trained_reg[name] = pipe

    pred = pipe.predict(Xr_test)

    reg_results.append({
        "Model": name,
        "MAE(log)": mean_absolute_error(yr_test, pred),
        "RMSE(log)": np.sqrt(mean_squared_error(yr_test, pred)),
        "R²(log)": r2_score(yr_test, pred)
    })

reg_results_df = pd.DataFrame(reg_results).sort_values("RMSE(log)")

print("\n=== REGRESSION MODEL RESULTS (log space) ===")
print(reg_results_df)

# Best regressor
best_reg_name = reg_results_df.iloc[0]["Model"]
best_reg_model = trained_reg[best_reg_name]
print("\nBest regressor:", best_reg_name)

# Regression plot: actual vs predicted
pred_log = best_reg_model.predict(Xr_test)

actual = np.expm1(yr_test)
pred = np.expm1(pred_log)

plt.figure(figsize=(6, 6))
plt.scatter(actual, pred, s=10)
plt.xlabel("Actual inundation_area_sqm")
plt.ylabel("Predicted inundation_area_sqm")
plt.title(f"Actual vs Predicted Inundation ({best_reg_name})")
plt.tight_layout()
plt.show()

# ==========================================================
# PART C — FEATURE IMPORTANCE / INTERPRETATION
# ==========================================================
print("\n=== FEATURE IMPORTANCE (TOP 15) ===")

model = best_cls_model.named_steps["model"]
feature_names = best_cls_model.named_steps["preprocessor"].get_feature_names_out()

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    feat_imp = feat_imp.sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(8, 5))
    plt.barh(feat_imp["feature"][::-1], feat_imp["importance"][::-1])
    plt.title("Top 15 Feature Importances (Flood Occurrence Model)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    print(feat_imp)

elif hasattr(model, "coef_"):  # Logistic Regression fallback
    coef = np.abs(model.coef_[0])
    feat_imp = pd.DataFrame({"feature": feature_names, "importance": coef})
    feat_imp = feat_imp.sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(8, 5))
    plt.barh(feat_imp["feature"][::-1], feat_imp["importance"][::-1])
    plt.title("Top 15 Logistic Regression Coefficients")
    plt.xlabel("|Coefficient|")
    plt.tight_layout()
    plt.show()

    print(feat_imp)

# ==========================================================
# PART D — RELIEF PRIORITY SYSTEM
# ==========================================================
df_out = X_test.copy()
df_out["flood_probability"] = best_prob
df_out["pred_inundation_area"] = 0.0

# Predict inundation only for likely flood areas
candidates = df_out["flood_probability"] >= 0.5
if candidates.any():
    pred_log_area = best_reg_model.predict(df_out.loc[candidates, X_reg.columns])
    df_out.loc[candidates, "pred_inundation_area"] = np.expm1(pred_log_area)

# Normalize vulnerability indicators
df_out["pop_norm"] = normalize(df_out["population_density_per_km2"])
df_out["infra_norm"] = normalize(df_out["infrastructure_score"])
df_out["hospital_norm"] = normalize(df_out["nearest_hospital_km"])
df_out["evac_norm"] = normalize(df_out["nearest_evac_km"])
df_out["inund_norm"] = normalize(df_out["pred_inundation_area"])

# Relief priority score
df_out["relief_priority_score"] = (
    0.30 * df_out["flood_probability"] +
    0.35 * df_out["inund_norm"] +
    0.20 * df_out["pop_norm"] +
    0.10 * (1 - df_out["infra_norm"]) +
    0.03 * df_out["hospital_norm"] +
    0.02 * df_out["evac_norm"]
)

# Priority levels
df_out["priority_level"] = pd.cut(
    df_out["relief_priority_score"],
    bins=[-1, 0.4, 0.7, 1.0],
    labels=["LOW", "MEDIUM", "HIGH"]
)

final_table = df_out[[
    "district", "latitude", "longitude",
    "flood_probability", "pred_inundation_area",
    "relief_priority_score", "priority_level"
]].sort_values("relief_priority_score", ascending=False)

print("\n=== TOP 20 RELIEF PRIORITY LOCATIONS ===")
print(final_table.head(20))

# Relief priority score map
plt.figure(figsize=(8, 6))
plt.scatter(final_table["longitude"], final_table["latitude"],
            c=final_table["relief_priority_score"], s=10)
plt.colorbar(label="Relief Priority Score")
plt.title("Sri Lanka Flood Relief Priority Map (Score)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()

# ==========================================================
# DISTRICT SUMMARY
# ==========================================================
district_summary = final_table.groupby("district").agg(
    total_locations=("relief_priority_score", "count"),
    avg_flood_probability=("flood_probability", "mean"),
    avg_pred_inundation=("pred_inundation_area", "mean"),
    avg_relief_score=("relief_priority_score", "mean"),
    max_relief_score=("relief_priority_score", "max"),
    high_priority_count=("priority_level", lambda x: (x == "HIGH").sum())
).reset_index()

district_summary = district_summary.sort_values("avg_relief_score", ascending=False)

print("\n=== DISTRICT SUMMARY (Top 10 Most Urgent Districts) ===")
print(district_summary.head(10))

# Top districts by high priority counts
top_high_districts = district_summary.sort_values("high_priority_count", ascending=False)
print("\n=== TOP DISTRICTS BY HIGH PRIORITY LOCATION COUNT ===")
print(top_high_districts.head(10))

# District bar plot (Top 10 avg relief score)
top10 = district_summary.head(10)
plt.figure(figsize=(10, 6))
plt.barh(top10["district"][::-1], top10["avg_relief_score"][::-1])
plt.xlabel("Average Relief Priority Score")
plt.title("Top 10 Districts by Average Relief Priority Score")
plt.tight_layout()
plt.show()

# ==========================================================
# DISTRICT-WISE MAP (Colored by District Average Score)
# ==========================================================
district_avg_map = district_summary.set_index("district")["avg_relief_score"].to_dict()
final_table["district_avg_relief"] = final_table["district"].map(district_avg_map)

plt.figure(figsize=(9, 7))
plt.scatter(final_table["longitude"], final_table["latitude"],
            c=final_table["district_avg_relief"], s=12)
plt.colorbar(label="District Average Relief Score")
plt.title("District-wise Flood Relief Priority Map (Avg Relief Score)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()

# ==========================================================
# DISTRICT PRIORITY CATEGORY MAP (LOW / MEDIUM / HIGH)
# ==========================================================
district_summary["district_priority_level"] = pd.cut(
    district_summary["avg_relief_score"],
    bins=[-1, 0.4, 0.7, 1.0],
    labels=["LOW", "MEDIUM", "HIGH"]
)

district_level_map = district_summary.set_index("district")["district_priority_level"].to_dict()
final_table["district_priority_level"] = final_table["district"].map(district_level_map)

level_code = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
final_table["district_priority_code"] = final_table["district_priority_level"].map(level_code)

plt.figure(figsize=(9, 7))
plt.scatter(final_table["longitude"], final_table["latitude"],
            c=final_table["district_priority_code"], s=12)

cbar = plt.colorbar()
cbar.set_ticks([0, 1, 2])
cbar.set_ticklabels(["LOW", "MEDIUM", "HIGH"])

plt.title("District-wise Flood Relief Priority Map (Priority Categories)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()

import joblib
joblib.dump(best_cls_model, "flood_occurrence_model.pkl")
joblib.dump(best_reg_model, "inundation_area_model.pkl")

