import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES

def main():
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Load data from Part A
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["delivery_hours"]
    
    # 2. Split data: 70% training (840 rows), temporary 30% (360 rows)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Split the 30% temporary into 180 validation and 180 test rows
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # 3. Scaling without leakage (fit scaler strictly on training rows only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    train_scaled_max_abs_mean = float(np.abs(X_train_scaled).max(axis=0).mean())
    test_scaled_max_abs_mean = float(np.abs(X_val_scaled).max(axis=0).mean())
    
    # 4. Baseline Model (Predict mean of y_train for every validation row)
    val_baseline_pred = np.full_like(y_val, y_train.mean())
    baseline_mae = mean_absolute_error(y_val, val_baseline_pred)
    baseline_mse = mean_squared_error(y_val, val_baseline_pred)
    baseline_rmse = np.sqrt(baseline_mse)
    baseline_r2 = r2_score(y_val, val_baseline_pred)
    
    # 5. Tree-depth diagnostic
    depths = [1, 2, 3, 4, 6, 8, 12, None]
    depth_results = []
    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        tr_rmse = np.sqrt(mean_squared_error(y_train, dt.predict(X_train)))
        va_rmse = np.sqrt(mean_squared_error(y_val, dt.predict(X_val)))
        depth_results.append((str(d) if d is not None else "unlimited", tr_rmse, va_rmse))
        
    depth_df = pd.DataFrame(depth_results, columns=["max_depth", "train_rmse", "val_rmse"])
    depth_df.to_csv("outputs/part_b_tree_depth.csv", index=False)
    
    # Find best tree depth based on lowest validation RMSE
    best_row = depth_df.loc[depth_df["val_rmse"].idxmin()]
    best_depth = int(best_row["max_depth"]) if best_row["max_depth"] != "unlimited" else None
    
    # 6. Four Candidate Models
    candidates = [
        ("linear_regression", Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])),
        ("decision_tree", DecisionTreeRegressor(max_depth=best_depth, random_state=42)),
        ("random_forest", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("gradient_boosting", GradientBoostingRegressor(learning_rate=0.1, random_state=42))
    ]
    
    metrics_rows = []
    # Add baseline model row first
    metrics_rows.append({
        "model": "mean_baseline", "split": "val",
        "MAE": round(baseline_mae, 4), "MSE": round(baseline_mse, 4),
        "RMSE": round(baseline_rmse, 4), "R2": round(baseline_r2, 4)
    })
    
    fitted_models = {}
    for name, model in candidates:
        model.fit(X_train, y_train)
        fitted_models[name] = model
        
        # Score on validation set
        preds = model.predict(X_val)
        metrics_rows.append({
            "model": name, "split": "val",
            "MAE": round(mean_absolute_error(y_val, preds), 4),
            "MSE": round(mean_squared_error(y_val, preds), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_val, preds)), 4),
            "R2": round(r2_score(y_val, preds), 4)
        })
        
    # 7. Learning-rate experiment for Gradient Boosting
    lr_results = []
    for lr in [0.01, 0.1, 1.0]:
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
        gb.fit(X_train, y_train)
        val_r2 = r2_score(y_val, gb.predict(X_val))
        val_rmse = np.sqrt(mean_squared_error(y_val, gb.predict(X_val)))
        lr_results.append((lr, val_rmse, val_r2))
        
    pd.DataFrame(lr_results, columns=["learning_rate", "val_rmse", "val_r2"]).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)
    
    # 8. Cross-validated comparison (5-fold CV on training + validation pool: 1,020 rows)
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_rows = []
    for name in ["linear_regression", "gradient_boosting"]:
        model = candidates[[c[0] for c in candidates].index(name)][1]
        cv_scores = cross_val_score(model, X_train_val, y_train_val, cv=kf, scoring="r2")
        cv_rows.append({
            "model": name,
            "mean_r2": round(cv_scores.mean(), 4),
            "std_r2": round(cv_scores.std(), 4)
        })
    pd.DataFrame(cv_rows).to_csv("outputs/part_b_cv.csv", index=False)
    
    # 9. Final Test scoring on winning model (Gradient Boosting)
    winning_model = fitted_models["gradient_boosting"]
    test_preds = winning_model.predict(X_test)
    metrics_rows.append({
        "model": "gradient_boosting", "split": "test",
        "MAE": round(mean_absolute_error(y_test, test_preds), 4),
        "MSE": round(mean_squared_error(y_test, test_preds), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, test_preds)), 4),
        "R2": round(r2_score(y_test, test_preds), 4)
    })
    
    # Save metrics table
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv("outputs/part_b_metrics.csv", index=False)
    
    # Save check metrics (scaler leakage check values)
    checks_data = [
        ("train_rows", len(X_train)),
        ("val_rows", len(X_val)),
        ("test_rows", len(X_test)),
        ("train_scaled_max_abs_mean", train_scaled_max_abs_mean),
        ("test_scaled_max_abs_mean", test_scaled_max_abs_mean),
    ]
    pd.DataFrame(checks_data, columns=["metric", "value"]).to_csv("outputs/part_b_checks.csv", index=False)
    
    print("Part B completed successfully! Check the outputs/ folder.")

if __name__ == "__main__":
    main()