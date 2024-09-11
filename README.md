# Clustering-with-K-means-PCA-Visualization
This project performs clustering on a dataset using a custom distance function and K-means algorithm. It includes dimensionality reduction with Principal Component Analysis (PCA) for visualizing the clusters in 2D. The main dataset used for this project is insurance.csv, or another CSV file specified by the user.

## Code Overview

- **distance(x, y)**: Calculates a combined distance between two points based on numerical and categorical attributes.
- **seuil(data)**: Determines a threshold to help in clustering decisions.
- **create_group(point)**: Creates a new group (cluster) starting with a single point.
- **Clusters(data)**: Determines the optimal number of clusters for the given data.
- **kmeans(X, k)**: Performs K-means clustering with `k` clusters.
- **PCA visualization**: Reduces data to 2 dimensions for visualizing the clusters using PCA.

## Results

After running the script:
- You will see the optimal number of clusters printed in the terminal.
- A 2D scatter plot will show the data points colored by their assigned cluster.

Here is the PCA plot showing the K-means clustering result:
![kmeans_clustering_result](https://github.com/user-attachments/assets/94426bd8-6e64-49eb-a9cc-8d6e8823c542)

