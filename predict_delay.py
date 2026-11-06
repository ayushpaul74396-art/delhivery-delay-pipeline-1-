import os
import joblib
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES

def main():
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Load inputs and artifacts
    new_raw = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    hub_segments = pd.read_csv("outputs/hub_segments.csv")
    
    impute_df = pd.read_csv("models/impute_values.csv")
    medians = dict(zip(impute_df["column"], impute_df["median_value"]))
    threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
    
    model = joblib.load("models/delay_model.pkl")
    
    # 2. Clean and build features for new shipments
    clean_new, _ = clean_shipments(new_raw, medians=medians)
    features_new = build_features(clean_new, hubs)
    
    # 3. Predict probabilities using ONLY the 17 official FEATURES
    probs = model.predict_proba(features_new[FEATURES])[:, 1]
    
    # 4. Assemble action list
    actions_df = pd.DataFrame({
        "shipment_id": new_raw["shipment_id"],
        "hub_id": features_new["hub_id"],
        "delay_probability": np.round(probs, 4)
    })
    
    actions_df = actions_df.merge(hub_segments[["hub_id", "segment"]], on="hub_id", how="left")
    actions_df["flagged"] = (actions_df["delay_probability"] >= threshold).astype(int)
    
    # Assign operations actions
    def assign_action(row):
        if row["flagged"] == 1 and row["segment"] == "Escalate":
            return "Escalate to hub manager"
        elif row["flagged"] == 1:
            return "Proactive customer call"
        else:
            return "No action"
            
    actions_df["action"] = actions_df.apply(assign_action, axis=1)
    
    # Final column ordering
    final_output = actions_df[[
        "shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"
    ]]
    final_output.to_csv("outputs/new_shipment_actions.csv", index=False)
    
    print("New shipment actions generated successfully in outputs/new_shipment_actions.csv!")

if __name__ == "__main__":
    main()
    