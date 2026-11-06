# Delhivery Last-Mile Delay Intelligence Pipeline

## How to run
To execute the pipeline from a fresh clone on the `main` branch, ensure Python and the required libraries are installed (`pip install -r requirements.txt`). Run the scripts in the following exact sequence from the repository root:
1. python generate_data.py
2. python part_a_clean_eda.py
3. python part_b_regression.py
4. python part_c_classification.py
5. python part_d_segments.py
6. python package_model.py
7. python predict_delay.py

*(Note: `models/delay_model.pkl` is regenerated automatically by running `package_model.py`)*

## Data findings
* **Service type delays:** **Economy** shipments show the lowest delay rate at 8.57% (27 delayed out of 315), while **Express** shipments experience the highest delay rate at 48.97% (119 delayed out of 243). **Standard** shipments sit in between at 20.56% (132 delayed out of 642).
* **Tier progression:** Delay rates increase significantly from Tier 1 hubs (4.00%) to Tier 2 hubs (21.50%) and Tier 3 hubs (44.00%).
* **Distance bands:** Delay rates scale monotonically across distance bands: 0–200 km (2.19%), 201–400 km (13.68%), 401–600 km (43.64%), and 601+ km (85.23%).
* **Top correlated features:** Excluding `delivery_hours` itself, the numeric columns most strongly correlated with delivery hours are `promised_hours` (0.837), `distance_km` (0.401), and the target indicator `is_delayed` (0.546).
* **Declared value transformation:** `declared_value_inr` was natural-log transformed (`log_declared_value`) because its mean (INR 1,533.05) and median (INR 1,121.50 Rupees) differed substantially, whereas its log counterpart normalized the distribution (mean 7.0280, median 7.0220).

## Model results (regression)
* **Baseline performance:** The mean baseline achieved a validation $R^2$ of $-0.021$, indicating total inability to track delivery hours.
* **Winning model:** Both **Linear Regression** and **Gradient Boosting** achieved top validation $R^2$ scores of 0.931, with Gradient Boosting winning out on final test evaluation ($R^2 = 0.949$).
* **Tree depth & learning rate:** Tree depth analysis showed high bias at depth 1 (train/val RMSE gap of 0.50) and high variance without limits (train/val RMSE gap of 7.79). For Gradient Boosting, a learning rate of `0.1` optimal ($R^2 = 0.931$), whereas `0.01` underfit ($R^2 = 0.713$) and `1.0` overshot ($R^2 = 0.844$).
* **Cross-validation:** 5-fold cross-validation confirmed stable performance across training and validation splits with mean $R^2$ values of 0.926 and 0.935 (standard deviation $\le 0.03$).

## Model results (classification)
* **Blind baseline:** The blind baseline (predicting all on-time) achieved an accuracy of 0.7667 but yielded zero True Positives (recall 0.0), rendering it useless for proactive intervention.
* **Feature selection:** Out of 17 input features, the 5 bookkeeping/noise columns (`label_print_batch`, `scanner_build_no`, `tape_roll_no`, `sticker_serial_mod`, `shelf_slot_no`) accounted for only 22.5% of total forest importance and all fell outside the top 7 F-score rankings. `SelectKBest` successfully isolated the top 7 operational features.
* **Tuning & Threshold:** Grid search and randomized search both converged on high accuracy ($0.8952$ and $0.8905$). Following the operations rule (minimizing missed delays), a probability threshold of **0.5** was chosen, achieving a validation recall of **0.756** and final test precision/recall/F1 of **0.689**, **0.756**, and **0.721** respectively.

## Operations recommendation
* **Hub Segmentation ($K=3$):** Using K-Means clustering, the 24 hubs were grouped into **Standard** (7 hubs, mean delay rate 2.57%), **Watch** (8 hubs, mean delay rate 15.50%), and **Escalate** (9 hubs, mean delay rate 46.00%).
* **Tier vs. Segment Mismatches:** Four hubs deviated from their official paper tier labels:
  * **HUB07 (Kolkata):** Labeled Tier 1 (Standard), but operates as **Watch**.
  * **HUB13 (Nagpur):** Labeled Tier 2 (Watch), but operates as **Escalate**.
  * **HUB15 (Kochi):** Labeled Tier 2 (Watch), but operates as **Escalate**.
  * **HUB22 (Dehradun):** Labeled Tier 3 (Escalate), but operates as **Watch**.
* **Action Protocol:** For the 48 new parcels, shipments flagged with a delay probability $\ge 0.5$ at an `Escalate` hub trigger an **Escalate to hub manager** action, while flagged shipments at `Standard` or `Watch` hubs trigger a **Proactive customer call**. All other parcels require **No action**.

## Production plan
1. **Performance & Reliability:** The pipeline runs fully offline using vectorized NumPy/Pandas operations and pre-trained scikit-learn pipelines, ensuring sub-second scoring speed for daily dispatch batches.
2. **Monitoring Metrics:** Operations must track weekly metrics with numerical triggers: (i) share of delivered parcels exceeding promised hours (alert if $> 25\%$), (ii) weekly share of parcels flagged for proactive calls (alert if $> 15\%$), and (iii) model recall measured against actual outcomes.
3. **Retraining Schedule:** Models must be retrained monthly, with mandatory retraining triggered immediately following major promotional windows (such as the 15–28 September sale event) to capture shifting logistics patterns.
4. **Data Drift:** Feature distributions (especially `distance_km` and `n_stops`) should be audited bi-weekly to detect network routing shifts or new lane openings.

## Limitations
* **Synthetic Data:** The dataset is synthetically generated rather than capturing live enterprise telemetry.
* **Test Set Size:** The test split contains a small absolute number of delayed parcels (41 in classification), leading to potential sample-size variance.
* **Threshold Sensitivity:** The decision threshold is tuned against a single validation split, meaning real-world cost trade-offs between unnecessary calls and missed delays may require dynamic adjustment.
