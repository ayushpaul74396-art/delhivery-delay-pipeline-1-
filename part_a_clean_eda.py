import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features

def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # 1. Load data
    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    
    raw_rows = len(raw)
    
    # Check service type spellings before cleaning
    service_spellings_before = raw["service_type"].nunique()
    
    # Check missing values before cleaning
    missing_distance_before = raw["distance_km"].isna().sum()
    missing_weight_before = raw["parcel_weight_kg"].isna().sum()
    sentinel_weight_rows = (raw["parcel_weight_kg"] > 100).sum()
    missing_value_before = raw["declared_value"].isna().sum()
    missing_pay_before = raw["payment_mode"].isna().sum()
    
    # 2. Clean shipments
    clean_df, medians = clean_shipments(raw)
    
    rows_after_dedup = len(clean_df)
    duplicate_rows_removed = raw_rows - rows_after_dedup
    
    service_spellings_after = clean_df["service_type"].nunique()
    
    # 3. Build features and merge hubs
    modelling_table = build_features(clean_df, hubs)
    rows_after_hub_merge = len(modelling_table)
    
    # Save modelling table
    modelling_table.to_csv("data/modelling_table.csv", index=False)
    
    # 4. Compute metrics for Part A checks
    nulls_in_features_and_target = modelling_table.isna().sum().sum()
    delayed_count = int(modelling_table["is_delayed"].sum())
    delay_rate = round(delayed_count / rows_after_hub_merge, 4)
    
    hours_q1 = round(modelling_table["delivery_hours"].quantile(0.25), 2)
    hours_q3 = round(modelling_table["delivery_hours"].quantile(0.75), 2)
    iqr = hours_q3 - hours_q1
    hours_lower_bound = round(hours_q1 - 1.5 * iqr, 2)
    hours_upper_bound = round(hours_q3 + 1.5 * iqr, 2)
    hours_outlier_count = int(((modelling_table["delivery_hours"] < hours_lower_bound) | 
                               (modelling_table["delivery_hours"] > hours_upper_bound)).sum())
    
    declared_value_mean_inr = round(clean_df["declared_value_inr"].mean(), 2)
    declared_value_median_inr = round(clean_df["declared_value_inr"].median(), 2)
    log_declared_value_mean = round(modelling_table["log_declared_value"].mean(), 4)
    log_declared_value_median = round(modelling_table["log_declared_value"].median(), 4)
    
    checks_data = [
        ("raw_rows", raw_rows),
        ("duplicate_rows_removed", duplicate_rows_removed),
        ("rows_after_dedup", rows_after_dedup),
        ("missing_distance_km_before", missing_distance_before),
        ("missing_parcel_weight_kg_before", missing_weight_before),
        ("sentinel_weight_rows", sentinel_weight_rows),
        ("missing_declared_value_before", missing_value_before),
        ("missing_payment_mode_before", missing_pay_before),
        ("service_type_spellings_before", service_spellings_before),
        ("service_type_spellings_after", service_spellings_after),
        ("declared_value_mean_inr", declared_value_mean_inr),
        ("log_declared_value_mean", log_declared_value_mean),
        ("median_distance_km", round(medians["distance_km"], 2)),
        ("median_parcel_weight_kg", round(medians["parcel_weight_kg"], 2)),
        ("median_declared_value_inr", round(medians["declared_value_inr"], 2)),
        ("rows_after_hub_merge", rows_after_hub_merge),
        ("nulls_in_features_and_target", nulls_in_features_and_target),
        ("delayed_count", delayed_count),
        ("delay_rate", delay_rate),
        ("hours_q1 / hours_q3", f"{hours_q1} / {hours_q3}"),
        ("hours_lower_bound / hours_upper_bound", f"{hours_lower_bound} / {hours_upper_bound}"),
        ("hours_outlier_count", hours_outlier_count),
        ("declared_value_median_inr", declared_value_median_inr),
        ("log_declared_value_median", log_declared_value_median),
    ]
    
    pd.DataFrame(checks_data, columns=["metric", "value"]).to_csv("outputs/part_a_checks.csv", index=False)
    
    # 5. Delay evidence table by dimensions
    groups = []
    
    # service_type group
    for st, group in modelling_table.groupby("service_type"):
        groups.append(("service_type", st, len(group), int(group["is_delayed"].sum()), 
                       round(group["is_delayed"].mean(), 4), round(group["delivery_hours"].mean(), 2)))
        
    # hub_tier group
    for ht, group in modelling_table.groupby("hub_tier"):
        groups.append(("hub_tier", str(ht), len(group), int(group["is_delayed"].sum()), 
                       round(group["is_delayed"].mean(), 4), round(group["delivery_hours"].mean(), 2)))
        
    # distance_band group
    bins = [0, 200, 400, 600, float("inf")]
    labels = ["0-200", "201-400", "401-600", "601+"]
    modelling_table["distance_band"] = pd.cut(modelling_table["distance_km"], bins=bins, labels=labels, right=True)
    for db, group in modelling_table.groupby("distance_band", observed=False):
        groups.append(("distance_band", str(db), len(group), int(group["is_delayed"].sum()), 
                       round(group["is_delayed"].mean(), 4), round(group["delivery_hours"].mean(), 2)))
        
    delay_group_df = pd.DataFrame(groups, columns=["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"])
    delay_group_df.to_csv("outputs/part_a_delay_by_group.csv", index=False)
    
    # 6. Correlations matrix
    numeric_cols = modelling_table.select_dtypes(include=[np.number]).columns
    corr_matrix = modelling_table[numeric_cols].corr().round(4)
    corr_matrix.to_csv("outputs/part_a_correlations.csv")
    
    print("Part A completed successfully! Check the outputs/ folder.")

if __name__ == "__main__":
    main()