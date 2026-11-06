import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from src.shipment_prep import FEATURES

def main():
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Load data
    df = pd.read_csv("data/modelling_table.csv")
    hubs = pd.read_csv("data/hubs.csv").sort_values("hub_id").reset_index(drop=True)
    
    # 2. Hub profile aggregations
    hub_profile = df.groupby("hub_id").agg(
        shipments=("delivery_hours", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean")
    ).reset_index()
    
    # Sort ascending by hub_id (HUB01 to HUB24)
    hub_profile = hubs[["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]].merge(hub_profile, on="hub_id")
    hub_profile.to_csv("outputs/hub_profile.csv", index=False)
    
    # 3. K-Means Segmentation (K=3)
    profile_cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(hub_profile[profile_cols])
    
    # Test separation across K = 2, 3, 4, 5
    sep_rows = []
    for k in [2, 3, 4, 5]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_matrix)
        
        # Calculate across-to-within distance ratio
        centers = kmeans.cluster_centers_
        # Simplified ratio representation matching brief expectations
        within_dist = np.mean([np.linalg.norm(scaled_matrix[labels == i] - centers[i]) for i in range(k)])
        across_dist = np.linalg.norm(centers.mean(axis=0) - centers)
        ratio = round(across_dist / (within_dist + 1e-5), 4)
        
        sizes = "-".join([str((labels == i).sum()) for i in range(k)])
        sep_rows.append({"K": k, "across_over_within": ratio, "cluster_sizes": sizes})
        
    pd.DataFrame(sep_rows).to_csv("outputs/part_d_separation.csv", index=False)
    
    # Final model with K=3
    kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    hub_profile["cluster"] = kmeans_3.fit_predict(scaled_matrix)
    
    # Map clusters to business segments based on delay rate ordering
    cluster_delay = hub_profile.groupby("cluster")["delay_rate"].mean().sort_values()
    cluster_mapping = {
        cluster_delay.index[0]: "Standard",
        cluster_delay.index[1]: "Watch",
        cluster_delay.index[2]: "Escalate"
    }
    hub_profile["segment"] = hub_profile["cluster"].map(cluster_mapping)
    
    # Save hub segments table
    hub_segments = hub_profile[["hub_id", "segment", "hub_city", "hub_tier"]]
    hub_segments.to_csv("outputs/hub_segments.csv", index=False)
    
    # Segment profile table
    seg_profile = hub_profile.groupby("segment").agg(
        hubs=("hub_id", "count"),
        hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        distance=("mean_distance_km", "mean"),
        stops=("mean_n_stops", "mean")
    ).reset_index().round(4)
    seg_profile.to_csv("outputs/segment_profile.csv", index=False)
    
    # Tier vs segment mismatch check
    tier_map = {1: "Standard", 2: "Watch", 3: "Escalate"}
    hub_profile["expected_segment"] = hub_profile["hub_tier"].map(tier_map)
    mismatches = hub_profile[hub_profile["segment"] != hub_profile["expected_segment"]]
    mismatches[["hub_id", "hub_city", "hub_tier", "segment"]].to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
    
    print("Part D Hub Segments completed successfully!")

if __name__ == "__main__":
    main()