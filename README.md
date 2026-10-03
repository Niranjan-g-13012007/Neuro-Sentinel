# NeuroSentinel

**Unsupervised Behavioral Pattern Discovery and Neural Anomaly Detection System**

A Streamlit application that discovers behavioral groups in network activity with **K-Means** and highlights unusual activity with a **neural-network autoencoder**, without using any labels. Every number, chart and flag in the app is computed from the supplied CSV when the app runs.

---

## 1. Project Overview

NeuroSentinel observes network activity records, groups similar behavior together (K-Means), learns how typical activity can be compressed and rebuilt (autoencoder), and flags activities the network cannot rebuild well. The two views are combined into a single, documented score and presented in an interactive dashboard with an investigation panel for individual activities.

## 2. Problem Statement

Network systems produce large volumes of behavioral data, but almost none of it is labeled. Labeling each activity as normal or anomalous by hand is slow, expensive and goes out of date as behavior changes.

## 3. Proposed Solution

Use **unsupervised learning** so that no labels are needed:

1. K-Means finds groups of similar behavior.
2. An autoencoder learns typical patterns; high reconstruction error means "does not fit the learned patterns".
3. The cluster context and the reconstruction error are combined into an interpretable result: *Normal Behavior* or *Potentially Unusual Behavior*.

The result is a prioritized list for human review, **not** a verdict that an activity is an attack.

## 4. System Architecture

```text
CSV dataset
   -> Data validation (columns, missing, duplicates, dtypes, infinities)
   -> Feature selection (activity_id excluded) + standardization check
   -> Exploratory analysis
   -> K-Means (elbow + silhouette) -> cluster summary -> PCA map
   -> Autoencoder (train/validation split) -> reconstruction error
   -> Percentile threshold from held-out validation errors
   -> Combined NeuroSentinel analysis (autoencoder + cluster distance)
   -> Dashboard + Anomaly Explorer
```

UI workflow: **Observe -> Discover -> Learn -> Detect -> Investigate.**

## 5. ML Methodology

### Dataset and assumptions

`data/neurosentinel_final_cleaned.csv` - 1,850 rows x 14 columns, inspected programmatically:

* `activity_id` is a unique integer identifier and is **not** a model feature.
* 13 numeric features: `duration_sec`, `src_bytes`, `dst_bytes`, `packets`, `connections`, `request_frequency`, `upload_ratio`, `night_activity`, `unique_destinations`, `protocol_HTTP`, `protocol_HTTPS`, `protocol_TCP`, `protocol_UDP`.
* No missing, duplicate or infinite values were found.
* **Already standardized:** every feature has mean ~ 0 and standard deviation ~ 1 (this includes the one-hot protocol and night-activity columns). The code verifies this at start-up (`check_standardization`) and **does not scale again**. A `StandardScaler` is only applied if the check fails on a different file.

Assumptions made:

* Because the values are z-scores, original units (bytes, seconds) cannot be recovered. All feature values in the app are z-scores (0 = dataset average).
* For display only, protocol names and the night-activity flag are decoded from their standardized columns (the larger of the two values per column corresponds to the original 1). The models still use the encoded columns.
* The dataset has no ground-truth labels, so **no accuracy, precision or recall is computed or claimed.**
* If a different CSV has extra numeric columns they are used as features; missing expected columns are skipped with a visible note; rows with missing/infinite values or repeated IDs are dropped and reported.

### K-Means

* K evaluated from 2 to 10 (`n_init=10`, fixed seed). Inertia gives the **elbow curve**; the **silhouette score** is the selection criterion.
* Selection rule: find the best silhouette, treat scores within 0.02 of it as a tie, and pick the **largest** K among the ties. A plain arg-max picks K=2 on this data (0.584), which only separates one small group from everything else; K=3 scores 0.576 and reveals two distinct groups. The rule and the alternatives are shown in the app, and K can be overridden in the sidebar.
* Reported: silhouette score, Davies-Bouldin index, inertia.
* Clusters are named neutrally ("Behavioral Cluster N"). A cluster is a group of similar behavior, not a verdict on intent; small clusters are never automatically labelled malicious.

### PCA

Two components are used for the interactive map only. Clustering and the autoencoder use all 13 features.

### Autoencoder

```text
13 -> Dense(8, ReLU) -> Dense(4, ReLU) -> Dense(2, ReLU) -> Dense(4, ReLU) -> Dense(8, ReLU) -> 13 (linear)
```

* Adam optimizer, mean-squared-error loss, batch size and maximum epochs from the sidebar.
* 80 % training / 20 % held-out validation (explicit split, fixed seed). `EarlyStopping` on validation loss (patience 10, best weights restored).
* The output layer is linear because standardized inputs contain negative values a ReLU output could never reproduce.

### Anomaly detection

* **Reconstruction error** `e` = mean squared difference between an activity and its reconstruction.
* **Threshold** = chosen percentile (default 95) of `e` on the **held-out validation set**, i.e. errors on data the network did not fit. Configurable in the sidebar.
* Autoencoder status: `Normal` / `Potential Anomaly`.

### Combined NeuroSentinel analysis

```text
r_AE = sqrt(e / tau_AE)         autoencoder RMSE relative to its threshold
r_CL = d / tau_cluster          distance to own centroid relative to that cluster's
                                percentile threshold (global fallback if cluster < 20 members)
S    = (r_AE + r_CL) / 2        equal-weight average
S > 1  ->  Potentially Unusual Behavior,  otherwise  Normal Behavior
```

Both ratios are in the same linear z-score scale (the square root converts squared error to RMSE). Equal weights are used because, without labels, there is no basis for trusting one signal more. Each activity also carries an *evidence* tag (`Both signals`, `Autoencoder only`, `Cluster distance only`, `Neither`) and a per-feature explanation (share of reconstruction error and deviation from the cluster average).

Because thresholds are percentiles, a roughly fixed share of activities is expected to be flagged **by construction**. The flags are for prioritizing review, not a measured detection rate.

## 6. Reference Run (default settings)

Produced by running the code on the supplied CSV (seed 42, TensorFlow 2.21 CPU, default sidebar settings). Results may differ slightly across platforms and TensorFlow versions.

| Item | Value |
| --- | --- |
| Silhouette by K (2..10) | 0.584, 0.576, 0.320, 0.413, 0.513, 0.518, 0.520, 0.413, 0.428 |
| Selected K (rule above) | 3 (silhouette 0.576, Davies-Bouldin 0.754) |
| Cluster sizes | 129 / 1,655 / 66 |
| Autoencoder | 100 epochs run (early stopping did not trigger), train loss 0.305, validation loss 0.318 |
| Reconstruction threshold (P95, validation) | 1.033 |
| Activities above the threshold (autoencoder) | 68 |
| Potentially Unusual Behavior (combined, S > 1) | 59 of 1,850 |

Cluster profiles (z-scores): Cluster 0 - high `request_frequency`, `connections`, `packets`, `unique_destinations`; Cluster 1 - close to the dataset average; Cluster 2 - high `src_bytes`, `duration_sec`, `upload_ratio`.

The loss was still decreasing slowly at epoch 100, so more epochs may help; this can be tried from the sidebar.

## 7. Technology Stack

Python, Streamlit, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Plotly.

## 8. Installation

Python 3.10-3.12 is recommended (TensorFlow 2.16+ is required for Python 3.12).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 9. How to Run

```bash
streamlit run app.py
```

The first start trains K-Means and the autoencoder (roughly 30-40 s on a CPU). Results are cached, so navigation and filtering are instant afterwards. In the sidebar, edit the controls and press **Apply settings**:

* **K** and the **percentile** update instantly (no retraining).
* **Epochs** and **batch size** retrain the autoencoder.

## 10. Project Structure

```text
NeuroSentinel/
├── app.py                    # entry point: sidebar, navigation
├── pipeline.py               # cached ML pipeline orchestration
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/config.toml    # light theme
├── data/neurosentinel_final_cleaned.csv
├── models/
│   ├── clustering.py         # K-Means, K selection, PCA, centroid distance
│   ├── autoencoder.py        # Keras autoencoder, training, reconstruction
│   └── anomaly_detection.py  # thresholds, combined score, per-activity explanation
├── utils/
│   ├── preprocessing.py      # loading, validation, standardization check
│   ├── visualization.py      # Plotly figures
│   └── ui.py                 # CSS, cards and Streamlit helpers
├── views/                    # one module per page
│   ├── overview.py  eda.py  clusters.py  anomaly.py  explorer.py  about.py
└── assets/
```

## 11. Screenshots

Add your own screenshots here after running the app.

* `assets/overview.png` - Overview dashboard
* `assets/clusters.png` - Behavioral Clusters
* `assets/anomaly.png` - Neural Anomaly Detection
* `assets/explorer.png` - Anomaly Explorer

## 12. Limitations

* No ground-truth labels: detection quality cannot be measured, and none is claimed.
* An anomaly is not an attack; unusual behavior can be benign. K-Means groups are descriptive, not malicious/benign classes.
* Percentile thresholds flag a roughly fixed share of data by design.
* The autoencoder trains on all activities, including any unusual ones, which can make it slightly more tolerant of them.
* Features are z-scores; original units are unavailable. Cluster-distance and reconstruction signals both depend on the scale of the standardized data.
* A 2-unit ReLU bottleneck is very compact; results depend on the seed and on training length.

## 13. Future Enhancements

* Evaluate on a separately labeled benchmark dataset (e.g., with injected or known attacks) to report real precision/recall.
* Compare against Isolation Forest, One-Class SVM and variational autoencoders.
* Train on a "clean" baseline period and score new data over time (streaming).
* SHAP-style feature attributions and per-cluster autoencoders.
* Persist trained models and export reports.

## 14. Testing performed

The application was run headless with Streamlit's `AppTest` across all six pages, with changed K, percentile, epochs, batch controls, and with every Anomaly Explorer filter and sort (including an empty-result search); no exceptions were raised. Retraining with identical settings gave identical reconstruction errors (maximum difference 0.0 on the test machine). The server was also started with `streamlit run app.py` and answered its health check. Visual appearance in a real browser was not screenshot-tested in this environment, so check the layout when you first run it.
