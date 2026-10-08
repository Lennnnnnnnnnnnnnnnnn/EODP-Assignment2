import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list, dendrogram
from EODP_A2_Preprocessing import df_rating


# same importing steps as in EODP_A2_FeatureSelection.py to ensure consistency in data preprocessing
cluster_data = df_rating.dropna(subset=['price']).copy()
cluster_data['bedrooms'] = cluster_data.groupby('room_type')['bedrooms'].transform(lambda x: x.fillna(x.median()))
cluster_data['bedrooms'] = cluster_data['bedrooms'].fillna(cluster_data['bedrooms'].median())

# This removes extreme outliers so that clustering is more accurate. As stated by Ten only 271 listings are above $1000
# Don't do this and you get weird groups
cluster_data = cluster_data[cluster_data['price'] < 1000]  

cluster_data = pd.get_dummies(cluster_data, columns=['room_type', 'property_group'], drop_first=False)

# chosen set 
features = ['price', 'accommodates', 'distance_from_cbd_km', 'room_type_Entire home/apt']

x_cluster = cluster_data[features].copy()
scaler = StandardScaler() # this is better
scaled_data = scaler.fit_transform(x_cluster)

# VAT to determine number of clusters #

# randomly sample 2000 listings so that VAT generation doesn't crash or take forever.
np.random.seed(3000)
sample_indices = np.random.choice(scaled_data.shape[0], size=2000, replace=False)
X_sample = scaled_data[sample_indices]

# calculate pairwise Euclidean distances for the sampled data
distance_matrix = pdist(X_sample, metric='euclidean')

# Visualize the VAT 
Z = linkage(distance_matrix, method='single')
ordered_indices = leaves_list(Z)

sq_dist_matrix = squareform(distance_matrix)
ordered_sq_matrix = sq_dist_matrix[ordered_indices, :][:, ordered_indices]


plt.figure(figsize=(8, 8))
plt.imshow(ordered_sq_matrix, cmap='gray', aspect='auto')
plt.colorbar(label='Euclidean Distance')
plt.title("VAT Heatmap of Sampled Listings")
plt.show()
# Suggests K = 3-4 clusters

# Elbow method to confirm k value #

# set range of k values to test
k_values = range(1, 11)
distortions = []

for k in k_values:
    # random_state is set for reproducibility
    kmeans = KMeans(n_clusters=k, random_state=3000)
    kmeans.fit(scaled_data)
    distortions.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, distortions, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Clusters = k')
plt.ylabel('Distortion')
plt.title('Elbow Method showing optimal k')
plt.xticks(k_values)
plt.grid(True, alpha=0.6)
plt.show()

# Showed k = 3 is a better choice than k = 4

# Final clustering and PCA visualization #

kmeans = KMeans(n_clusters=3, random_state=3000)
cluster_data['kmeans_cluster'] = kmeans.fit_predict(scaled_data)

pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)
cluster_data['pca1'] = pca_components[:, 0]
cluster_data['pca2'] = pca_components[:, 1]

loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features)
print("PCA Loadings:\n", loadings)
print(f"\nVariance explained by PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"Variance explained by PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x = 'pca1',
    y = 'pca2',
    hue = 'kmeans_cluster', 
    data = cluster_data, 
    palette='Set1', 
    alpha=0.7, 
    edgecolor=None
)

plt.title('PCA of Airbnb Listings with K-Means Clusters')
plt.xlabel(f'PC1 (Size & Cost Axis)')
plt.ylabel(f'PC2 (Distance Axis)')
plt.legend(title='K-Means Clusters', loc='best')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

# Hierarchical clustering and comparison with K-Means #
# This section was done with the help of Generative AI (Gemini) #
np.random.seed(3000)
index = np.random.choice(len(scaled_data), size=2000, replace=False)
Z_ward = linkage(scaled_data[index], method='ward')

# Create a dendrogram to visualize the hierarchical clustering
plt.figure(figsize=(12, 6))
dendrogram(Z_ward, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=12., show_contracted=True)
plt.title('Hierarchical Clustering Dendrogram (Ward\'s Method)')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()


cluster_data['hierarchical_cluster'] = fcluster(linkage(scaled_data, method='ward'), t=3, criterion='maxclust')
sample = cluster_data.iloc[index].copy()
sample['hierarchical_cluster'] = fcluster(Z_ward, t=3, criterion='maxclust')

# Compare K-Means and Hierarchical clustering results visually
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
sns.scatterplot(x='pca1', y='pca2', hue='kmeans_cluster', data=sample,
                palette='Set1', alpha=0.7, ax=axes[0])
axes[0].set_title('K-Means (k=3)')
sns.scatterplot(x='pca1', y='pca2', hue='hierarchical_cluster', data=sample,
                palette='Set1', alpha=0.7, ax=axes[1])
axes[1].set_title('Ward Hierarchical (k=3)')
plt.tight_layout()
plt.show()

# Mean profiles of clusters
cluster_profiles = cluster_data.groupby('kmeans_cluster')[features].mean()
hierarchical_profiles = cluster_data.groupby('hierarchical_cluster')[features].mean()

fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(
    cluster_data['pca1'], cluster_data['pca2'],
    c=cluster_data['price'], cmap='viridis',
    alpha=0.6, s=15, edgecolor='none'
)
plt.colorbar(sc, label='price')
ax.set_xlabel(f'PC1: Size & Cost')
ax.set_ylabel(f'PC2: Distance from CBD')
ax.set_title('PCA coloured by price')
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

print("\nAverage Property Profile per Cluster")
print(cluster_profiles)
print("\nAverage Property Profile per Hierarchical Cluster")
print(hierarchical_profiles)
