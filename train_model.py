# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib
import numpy as np
from feature_extraction import extract_features_from_dataframe, extract_features_for_prediction

# -----------------------------------------------
# Load Dataset
# -----------------------------------------------
dataset_path = "dataset/dataset.csv"
print("Loading dataset...")
data = pd.read_csv(dataset_path)
print(f"Dataset loaded. Total samples: {len(data)}")

# -----------------------------------------------
# Check class distribution BEFORE training
# This is needed to balance XGBoost correctly
# -----------------------------------------------
data.columns = data.columns.str.strip().str.lower()
label_cols = [col for col in data.columns if "label" in col]
label_column = label_cols[0]
y_raw = data[label_column].astype(str).str.lower().str.strip()
y_check = y_raw.map({"good": 0, "bad": 1})
num_legitimate = int((y_check == 0).sum())
num_phishing = int((y_check == 1).sum())
print(f"\nClass distribution:")
print(f"  Legitimate (good): {num_legitimate}")
print(f"  Phishing   (bad) : {num_phishing}")

# FIXED: scale_pos_weight tells XGBoost how to balance classes
# Without this XGBoost was predicting phishing for everything
scale_pos_weight = num_legitimate / num_phishing
print(f"  XGBoost scale_pos_weight: {scale_pos_weight:.4f}")

# -----------------------------------------------
# Extract Features
# -----------------------------------------------
print("\nExtracting features...")
X, y = extract_features_from_dataframe(data)
print(f"Features extracted. Shape: {X.shape}")

# Replace any NaN values just in case
nan_count = np.isnan(X).sum()
if nan_count > 0:
    print(f"WARNING: {nan_count} NaN values found — replacing with 0")
    X = np.nan_to_num(X, nan=0.0)
else:
    print("No NaN values found. Data is clean.")

# -----------------------------------------------
# Split Data
# -----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# -----------------------------------------------
# Train Random Forest
# -----------------------------------------------
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight={0: 1, 1: 1.5},
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\n--- Random Forest Results ---")
print(f"Accuracy: {accuracy_score(y_test, rf_pred)*100:.2f}%")
print(classification_report(y_test, rf_pred, target_names=["Legitimate", "Phishing"]))

# Sanity check on known legitimate URLs
print("--- Random Forest Sanity Check ---")
test_urls_legit = [
    "https://www.google.com",
    "https://www.youtube.com",
    "https://www.amazon.co.uk",
    "https://www.bbc.co.uk",
    "https://www.microsoft.com",
    "https://www.github.com",
    "https://www.wikipedia.org",
]
test_urls_phishing = [
    "http://paypal-secure.update-account.com/signin",
    "http://192.168.1.1/banking/login/confirm",
    "http://ebayisapi.com/verify-account?cmd=_login",
]
print("  Legitimate URLs (should all show Safe):")
for u in test_urls_legit:
    f = extract_features_for_prediction(u)
    pred = rf_model.predict(f)[0]
    prob = rf_model.predict_proba(f)[0]
    label = "Safe" if pred == 0 else "PHISHING -- WRONG"
    print(f"    {u[:55]:<55} -> {label} ({int(max(prob)*100)}%)")

print("  Phishing URLs (should all show Phishing):")
for u in test_urls_phishing:
    f = extract_features_for_prediction(u)
    pred = rf_model.predict(f)[0]
    prob = rf_model.predict_proba(f)[0]
    label = "Phishing" if pred == 1 else "Safe -- WRONG"
    print(f"    {u[:55]:<55} -> {label} ({int(max(prob)*100)}%)")

# 10-Fold Cross Validation
print("\nRunning 10-Fold Cross Validation for Random Forest...")
rf_cv = cross_val_score(rf_model, X, y, cv=10, scoring='accuracy', n_jobs=-1)
print(f"CV Accuracy: {rf_cv.mean()*100:.2f}% (+/- {rf_cv.std()*100:.2f}%)")

# -----------------------------------------------
# Train XGBoost
# -----------------------------------------------
print("\nTraining XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    # FIXED: Added scale_pos_weight — this is what was missing before.
    # Without this XGBoost was biased toward predicting phishing for everything.
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
xgb_pred = xgb_model.predict(X_test)

print("\n--- XGBoost Results ---")
print(f"Accuracy: {accuracy_score(y_test, xgb_pred)*100:.2f}%")
print(classification_report(y_test, xgb_pred, target_names=["Legitimate", "Phishing"]))

# Sanity check on known legitimate URLs
print("--- XGBoost Sanity Check ---")
print("  Legitimate URLs (should all show Safe):")
for u in test_urls_legit:
    f = extract_features_for_prediction(u)
    pred = xgb_model.predict(f)[0]
    prob = xgb_model.predict_proba(f)[0]
    label = "Safe" if pred == 0 else "PHISHING -- WRONG"
    print(f"    {u[:55]:<55} -> {label} ({int(max(prob)*100)}%)")

print("  Phishing URLs (should all show Phishing):")
for u in test_urls_phishing:
    f = extract_features_for_prediction(u)
    pred = xgb_model.predict(f)[0]
    prob = xgb_model.predict_proba(f)[0]
    label = "Phishing" if pred == 1 else "Safe -- WRONG"
    print(f"    {u[:55]:<55} -> {label} ({int(max(prob)*100)}%)")

# 10-Fold Cross Validation
print("\nRunning 10-Fold Cross Validation for XGBoost...")
xgb_cv = cross_val_score(xgb_model, X, y, cv=10, scoring='accuracy', n_jobs=-1)
print(f"CV Accuracy: {xgb_cv.mean()*100:.2f}% (+/- {xgb_cv.std()*100:.2f}%)")

# -----------------------------------------------
# Save Models
# -----------------------------------------------
joblib.dump(rf_model, "phishing_model_rf.pkl")
print("\nRandom Forest model saved as phishing_model_rf.pkl")

joblib.dump(xgb_model, "phishing_model_xgb.pkl")
print("XGBoost model saved as phishing_model_xgb.pkl")

print("\nTraining complete. Both models saved successfully.")
print("\nIMPORTANT REMINDER:")
print("  1. Make sure you deleted your OLD .pkl files before running app.py")
print("  2. Run: python app.py")
print("  3. Test with google.com, youtube.com — they should show as Safe")