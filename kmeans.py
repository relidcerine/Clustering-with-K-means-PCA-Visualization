import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv('d.csv')

# Set a random seed for reproducibility
np.random.seed(800)

# Function to calculate the distance between two points (x and y)
def distance(x, y):
  num_dims = []
  cat_dims = []
  
  for col in range(len(x)):
    if isinstance(x[col], (int, float)):
      num_dims.append(col)
    else:
      cat_dims.append(col)

  # Calculate Euclidean distance for numerical dimensions
  num_dist = np.linalg.norm(x[num_dims] - y[num_dims])

  # Calculate distance for categorical dimensions (proportion of different categories)
  cat_dist = sum([x[cat_dim] != y[cat_dim] for cat_dim in cat_dims])/len(cat_dims)
  
  # Calculate the total distance by combining numerical and categorical distances
  dist = np.sqrt(num_dist ** 2 + cat_dist ** 2)
  return dist

# Function to calculate a threshold for grouping points
def seuil(data):
      
  # Extract only the numerical data from the dataset
  matrix = dataset.select_dtypes(include=['int', 'float']).values
  
  # Calculate the distance matrix (pairwise Euclidean distances)
  dist_matrix = np.sqrt(np.sum((matrix[:, None] - matrix) ** 2, axis=2))
  
  # Calculate the mean and standard deviation of distances
  mean_dist = np.mean(dist_matrix)
  std_dist = np.std(dist_matrix)

  # Calculate a threshold using the mean and standard deviation
  seuil = mean_dist + 2 * std_dist
 
  return seuil*1.2

# Function to create a new group with a single point
def create_group(point):
    return [point]

# Function to find the optimal number of clusters in the dataset
def Clusters(data):
  center = data[np.random.choice(data.shape[0])]
  grps = []
  grp = create_group(center)

  for i in data:
    d = distance(center, i)
    if d <= seuil(data) :
      grp.append(i)
    else :
      grps.append(create_group(i))
    i +=i
  return len(grps)

# Function to perform k-means clustering on the dataset
def kmeans(X, k):
      
    enc = OrdinalEncoder()
    X = pd.DataFrame(enc.fit_transform(X), columns=X.columns)
    X = X.values
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    while True:
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        if np.all(centroids == new_centroids):
            break
          
        centroids = new_centroids
        
    return centroids, labels

k = Clusters(dataset.values)
print('The optimal number of clusters is: ',k)

centroids, labels = kmeans(dataset, k)
dataset['label'] = labels

enc = OrdinalEncoder()
dataset = pd.DataFrame(enc.fit_transform(dataset), columns=dataset.columns)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(dataset)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
plt.title('K-means clustering')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()