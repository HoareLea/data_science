import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA

# Clustering
"""
Overview

Clustering is an unsupervised learning technique used to group data points so that 
items in the same cluster are more similar to each other than to those in other clusters. 
It’s often used for exploring structure in data, customer segmentation, anomaly detection, 
or as a preprocessing step for other models. Unlike classification/regression, there is no 
“y” during training - only X and a notion of similarity.


Common families:
1. Centroid-based (e.g., K-Means)
   - Assumes clusters are roughly spherical around a centroid.
   - Fast and scalable.
   - You choose k (number of clusters).
   - Objective: minimise within-cluster sum of squares (inertia).
   - Sensitive to scale and outliers.
2. Hierarchical (e.g., Agglomerative)
   - Builds a tree (dendrogram) by iteratively merging or splitting clusters.
   - Doesn’t require choosing k upfront if you cut the tree later, but you often set n_clusters.
   - Choice of linkage matters (ward, average, complete, single).
   - Can capture nested/grouped structure.
3. Density-based (e.g., DBSCAN, HDBSCAN)
   - Clusters are dense regions separated by low-density regions.
   - Finds arbitrarily shaped clusters.
   - Flags outliers as noise.
   - Requires distance scale tuning (eps, min_samples) and benefits from standardised features.


How to judge quality (since there’s no ground truth)?
- Internal metrics:
  * Silhouette score ([-1, 1], higher is better).
  * Davies–Bouldin score (>=0, lower is better).
- Stability / interpretability:
  * Are results robust to small changes?
  * Do clusters make sense to domain experts?
"""

# Load the dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Keep a copy of the target for post-hoc evaluation (not used to fit clusters)
y_true = df["Survived"].values

# Drop non-numeric / high-cardinality text fields and identifiers
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch"])

# Preprocessing
categorical_cols = ["Sex", "Embarked"]
numeric_cols = [c for c in df.columns if c not in categorical_cols + ["Survived"]]

# Handle any missing values with simple imputation inside the pipeline (via 'passthrough' approach)
# Using pandas fillna before pipeline for simplicity and reproducibility:
for c in numeric_cols:
    df[c] = df[c].fillna(df[c].median())
for c in categorical_cols:
    df[c] = df[c].fillna(df[c].mode().iloc[0])

X = df.drop(columns=["Survived"])

# Scale numeric columns
# One-hot encode categoricals
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ],
    remainder="drop",
)

# Fit transform to get the full feature space (this is what we cluster on)
X_full = preprocess.fit_transform(X)

# Optional: a 2D projection for plotting if you want to add visuals later (not used for clustering)
# pca2 = PCA(n_components=2, random_state=72)
# X_2d = pca2.fit_transform(X_full)

# Helper: evaluate clusters
def evaluate_clustering(X_space, labels, name, y_true=None):
    unique_labels = np.unique(labels)
    valid_for_internal = len(unique_labels) > 1 and not (len(unique_labels) == 1 and unique_labels[0] == -1)

    print(f"\n=== {name} ===")
    print(f"Number of clusters (excluding noise): {len([u for u in unique_labels if u != -1])}")
    if valid_for_internal:
        sil = silhouette_score(X_space, labels)
        db = davies_bouldin_score(X_space, labels)
        print(f"Silhouette score: {sil:.3f}  (higher is better, max 1.0)")
        print(f"Davies–Bouldin:  {db:.3f}  (lower is better)")
    else:
        print("Internal metrics not computed (likely single cluster or all noise).")

    if y_true is not None:
        ari = adjusted_rand_score(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels)
        print(f"Adjusted Rand Index vs. Survived: {ari:.3f} (1.0 perfect, 0 ~ random)")
        print(f"NMI vs. Survived:                {nmi:.3f} (1.0 perfect, 0 ~ random)")


# Fit & compare algorithms
# 1) K-Means (k=2)
kmeans = KMeans(n_clusters=3, n_init=10, random_state=72)
kmeans_labels = kmeans.fit_predict(X_full)
evaluate_clustering(X_full, kmeans_labels, "K-Means (k=3)", y_true=y_true)

# 2) Agglomerative (Ward linkage)
agglo = AgglomerativeClustering(n_clusters=3, linkage="ward")
agglo_labels = agglo.fit_predict(X_full.toarray() if hasattr(X_full, "toarray") else X_full)
evaluate_clustering(X_full, agglo_labels, "Agglomerative (n_clusters=3, ward)", y_true=y_true)

# 3) DBSCAN (parameters are sensitive to scale & dimensionality; tweak as needed)
dbscan = DBSCAN(eps=1.1, min_samples=10)  # try 1.0–3.0 depending on results
dbscan_labels = dbscan.fit_predict(X_full)
evaluate_clustering(X_full, dbscan_labels, "DBSCAN (eps=1.1, min_samples=10)", y_true=y_true)

# Quick K-Means elbow sweep
print("\n=== K-Means elbow check (inertia) ===")
inertias = []
Ks = list(range(2, 9))
for k in Ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=72)
    km.fit(X_full)
    inertias.append(km.inertia_)
    print(f"k={k}: inertia={km.inertia_:.2f}")

# A simple line chart of Ks vs. inertias helps eyeball the elbow.