import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS

def main():
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Load data from Part A
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]
    
    # Stratified split: 70% training, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # 2. Blind baseline and three classifiers
    def evaluate_preds(y_true, y_pred, model_name):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return {
            "model": model_name, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 4)
        }
        
    models_rows = []
    
    # Blind baseline (always predict 0 / not delayed)
    baseline_preds = np.zeros_like(y_val)
    models_rows.append(evaluate_preds(y_val, baseline_preds, "blind_all_on_time"))
    
    # Logistic Regression pipeline
    log_reg = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(random_state=42))])
    log_reg.fit(X_train, y_train)
    models_rows.append(evaluate_preds(y_val, log_reg.predict(X_val), "logistic_regression"))
    
    # Decision Tree (find best depth for F1)
    best_dt_f1, best_dt_depth = -1, 3
    for d in range(2, 11):
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        f1 = f1_score(y_val, dt.predict(X_val), zero_division=0)
        if f1 > best_dt_f1:
            best_dt_f1 = f1
            best_dt_depth = d
            
    dt_best = DecisionTreeClassifier(max_depth=best_dt_depth, random_state=42)
    dt_best.fit(X_train, y_train)
    models_rows.append(evaluate_preds(y_val, dt_best.predict(X_val), f"decision_tree_{best_dt_depth}"))
    
    # Default Random Forest
    rf_default = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_default.fit(X_train, y_train)
    models_rows.append(evaluate_preds(y_val, rf_default.predict(X_val), "random_forest_default"))
    
    pd.DataFrame(models_rows).to_csv("outputs/part_c_models.csv", index=False)
    
    # 3. Feature selection & noise analysis
    selector_full = SelectKBest(f_classif, k="all")
    selector_full.fit(X_train, y_train)
    
    feature_scores_rows = []
    for col, score in zip(FEATURES, selector_full.scores_):
        feature_scores_rows.append({
            "feature": col,
            "f_score": round(score, 4),
            "rf_importance": round(rf_default.feature_importances_[FEATURES.index(col)], 4),
            "is_noise": col in NOISE_COLUMNS
        })
    feature_scores_df = pd.DataFrame(feature_scores_rows).sort_values(by="f_score", ascending=False)
    feature_scores_df.to_csv("outputs/part_c_feature_scores.csv", index=False)
    
    # Select top 7 features
    selector_k7 = SelectKBest(f_classif, k=7)
    selector_k7.fit(X_train, y_train)
    top_7_features = [FEATURES[i] for i in selector_k7.get_support(indices=True)]
    pd.DataFrame({"selected_feature": top_7_features}).to_csv("outputs/part_c_selectk.csv", index=False)
    
    # Transform training/validation data to top 7 features
    X_train_k7 = selector_k7.transform(X_train)
    X_val_k7 = selector_k7.transform(X_val)
    
    # 4. Hyperparameter tuning (GridSearchCV & RandomizedSearchCV)
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10]
    }
    rf_tune = RandomForestClassifier(random_state=42)
    
    grid_search = GridSearchCV(rf_tune, param_grid, cv=5, scoring="f1")
    grid_search.fit(X_train_k7, y_train)
    
    rand_search = RandomizedSearchCV(rf_tune, param_grid, n_iter=9, cv=5, random_state=42, scoring="f1")
    rand_search.fit(X_train_k7, y_train)
    
    # Validation F1 scores for tuning comparison
    grid_val_f1 = f1_score(y_val, grid_search.predict(X_val_k7), zero_division=0)
    rand_val_f1 = f1_score(y_val, rand_search.predict(X_val_k7), zero_division=0)
    default_val_f1 = f1_score(y_val, rf_default.predict(selector_full.transform(X_val)), zero_division=0)
    
    tuning_rows = [
        {
            "search": "grid_search", "candidates": 27, "cv_folds": 5, "total_fits": 135,
            "best_params": str(grid_search.best_params_), "best_cv_accuracy": round(grid_search.best_score_, 4),
            "val_f1": round(grid_val_f1, 4)
        },
        {
            "search": "randomized_search", "candidates": 9, "cv_folds": 5, "total_fits": 45,
            "best_params": str(rand_search.best_params_), "best_cv_accuracy": round(rand_search.best_score_, 4),
            "val_f1": round(rand_val_f1, 4)
        },
        {
            "search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 0,
            "best_params": "default", "best_cv_accuracy": "",
            "val_f1": round(default_val_f1, 4)
        }
    ]
    pd.DataFrame(tuning_rows).to_csv("outputs/part_c_tuning.csv", index=False)
    
    # 5. Threshold search & Operations Rule (Recall >= 0.80)
    final_model = grid_search.best_estimator_
    val_probs = final_model.predict_proba(X_val_k7)[:, 1]
    
    thresholds = np.round(np.arange(0.1, 1.0, 0.1), 1)
    threshold_rows = []
    chosen_threshold = 0.5
    
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        rec = recall_score(y_val, preds, zero_division=0)
        prec = precision_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        threshold_rows.append({"threshold": t, "recall": round(rec, 4), "precision": round(prec, 4), "f1": round(f1, 4)})
        
    pd.DataFrame(threshold_rows).to_csv("outputs/part_c_thresholds.csv", index=False)
    
    # Find highest threshold with recall >= 0.80 (defaults to 0.5 if none meet rule)
    valid_thresholds = [row["threshold"] for row in threshold_rows if row["recall"] >= 0.80]
    if valid_thresholds:
        chosen_threshold = max(valid_thresholds)
        
    # 6. Final test evaluation
    X_test_k7 = selector_k7.transform(X_test)
    test_probs = final_model.predict_proba(X_test_k7)[:, 1]
    test_preds = (test_probs >= chosen_threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds, labels=[0, 1]).ravel()
    final_test_row = {
        "threshold": chosen_threshold, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": round(accuracy_score(y_test, test_preds), 4),
        "precision": round(precision_score(y_test, test_preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, test_preds, zero_division=0), 4),
        "f1": round(f1_score(y_test, test_preds, zero_division=0), 4)
    }
    pd.DataFrame([final_test_row]).to_csv("outputs/part_c_final_test.csv", index=False)
    
    # Save final config reference
    config_row = {
        "selector_k": 7, "n_estimators": grid_search.best_params_["n_estimators"],
        "max_depth": grid_search.best_params_["max_depth"], "min_samples_leaf": grid_search.best_params_["min_samples_leaf"],
        "threshold": chosen_threshold, "chosen_from": "grid_search"
    }
    pd.DataFrame([config_row]).to_csv("outputs/part_c_final_config.csv", index=False)
    
    print("Part C completed successfully! Check the outputs/ folder.")

if __name__ == "__main__":
    main()