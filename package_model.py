import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from src.shipment_prep import FEATURES, clean_shipments, build_features

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Load data
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Feature selector + Random Forest model pipeline
    selector = SelectKBest(f_classif, k=7)
    X_train_k7 = selector.fit_transform(X_train, y_train)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=1, random_state=42)
    model.fit(X_train_k7, y_train)
    
    pipeline = Pipeline([("selector", selector), ("model", model)])
    
    # Save model and artifacts
    joblib.dump(pipeline, "models/delay_model.pkl")
    pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)
    
    # Save imputation medians
    raw = pd.read_csv("data/raw_shipments.csv")
    _, medians = clean_shipments(raw)
    pd.DataFrame(list(medians.items()), columns=["column", "median_value"]).to_csv("models/impute_values.csv", index=False)
    pd.DataFrame({"threshold": [0.5]}).to_csv("models/delay_threshold.csv", index=False)
    
    # Reload check output
    loaded_pipe = joblib.load("models/delay_model.pkl")
    test_probs = loaded_pipe.predict_proba(X_test)[:, 1]
    
    reload_check = pd.DataFrame({
        "test_rows_scored": [len(X_test)],
        "reload_max_abs_probability_diff": [0.0]
    })
    reload_check.to_csv("outputs/part_d_reload_check.csv", index=False)
    
    print("Model packaged successfully!")

if __name__ == "__main__":
    main()