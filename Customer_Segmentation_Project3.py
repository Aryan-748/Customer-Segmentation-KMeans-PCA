import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Load Dataset

df = pd.read_csv("Mall_Customers.csv")

print("\nFirst 5 Rows\n")
print(df.head())

print("\nDataset Info\n")
print(df.info())

print("\nStatistical Summary\n")
print(df.describe())

print("\nMissing Values\n")
print(df.isnull().sum())


# Feature Selection

X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

print("\nSelected Features\n")
print(X.head())


# Standardization

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled Data (First 5 Rows)\n")
print(X_scaled[:5])


# PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA Output (First 5 Rows)\n")
print(X_pca[:5])

print("\nExplained Variance Ratio\n")
print(pca.explained_variance_ratio_)


# Elbow Method

wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_pca)

    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))

plt.plot(
    range(1,11),
    wcss,
    marker='o'
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.show()


# Silhouette Score

print("\nSilhouette Scores\n")

for k in range(2,11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_pca)

    score = silhouette_score(
        X_pca,
        labels
    )

    print(
        f"K={k}  Silhouette Score={score:.3f}"
    )


# Final KMeans

optimal_k = 5

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(
    X_pca
)

df["Cluster"] = clusters


# Customer Segmentation Plot

plt.figure(figsize=(10,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters,
    cmap="viridis"
)

plt.title(
    "Customer Segmentation using K-Means"
)

plt.xlabel(
    "PCA Component 1"
)

plt.ylabel(
    "PCA Component 2"
)

plt.show()


# Cluster Summary

cluster_summary = df.groupby(
    "Cluster"
).mean(
    numeric_only=True
)

print("\nCluster Summary\n")

print(cluster_summary)


# Save Output

df.to_csv(
    "customer_segments.csv",
    index=False
)

print(
    "\nCustomer Segmentation Dataset Saved!"
)