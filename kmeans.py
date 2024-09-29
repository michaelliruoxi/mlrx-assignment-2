import numpy as np

class KMeansCustom:
    def __init__(self, n_clusters=3, init_method='random', max_iter=300, tol=1e-4, manual_centroids=None):
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.manual_centroids = manual_centroids  # Store manually selected centroids if provided

    def initialize_centroids(self, X):
        if self.init_method == 'random':
            self.centroids = self._initialize_random(X)
            print(self.centroids)
        elif self.init_method == 'farthest_first':
            self.centroids = self._initialize_farthest_first(X)
        elif self.init_method == 'kmeans++':
            self.centroids = self._initialize_kmeans_plus_plus(X)
        elif self.init_method == 'manual':
            self.centroids = np.array(self.manual_centroids)  # Use the provided manual centroids  
            print(self.centroids)
        return self.centroids

    def _initialize_random(self, X):
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def _initialize_farthest_first(self, X):
        centroids = [X[np.random.randint(X.shape[0])]]
        for _ in range(1, self.n_clusters):
            dist_sq = np.array([min(np.linalg.norm(x - c)**2 for c in centroids) for x in X])
            next_centroid = X[np.argmax(dist_sq)]
            centroids.append(next_centroid)
        return np.array(centroids)

    def _initialize_kmeans_plus_plus(self, X):
        centroids = [X[np.random.randint(X.shape[0])]]
        for _ in range(1, self.n_clusters):
            dist_sq = np.array([min(np.linalg.norm(x - c)**2 for c in centroids) for x in X])
            probs = dist_sq / dist_sq.sum()
            cumulative_probs = np.cumsum(probs)
            r = np.random.rand()
            next_centroid = X[np.where(cumulative_probs >= r)[0][0]]
            centroids.append(next_centroid)
        return np.array(centroids)

    def fit(self, X):
        if self.centroids is None:
            self.initialize_centroids(X)  # Ensure centroids are initialized before proceeding
        for _ in range(self.max_iter):
            closest_centroids = self._assign_clusters(X)
            
            new_centroids = np.array([X[closest_centroids == k].mean(axis=0) for k in range(self.n_clusters)])

            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                break

            self.centroids = new_centroids

    """def _assign_clusters(self, X):
        if self.centroids is None:
            raise ValueError("Centroids are not initialized.")  # Ensure centroids are initialized
        distances = np.array([np.linalg.norm(X - c, axis=1) for c in self.centroids])
        return np.argmin(distances, axis=0)"""
    

    
    def _assign_clusters(self, X):
        # Ensure centroids are initialized
        if self.centroids is None:
            raise ValueError("Centroids are not initialized.")
        
        # Calculate distances from each point to each centroid
        distances = np.array([np.linalg.norm(X - c, axis=1) for c in self.centroids])
        
        # Assign each point to the nearest centroid
        assigned_clusters = np.argmin(distances, axis=0)
        
        # Count how many points are assigned to each cluster
        points_per_cluster = np.bincount(assigned_clusters, minlength=self.n_clusters)
        
        # Check for empty clusters
        for k in range(self.n_clusters):
            if points_per_cluster[k] == 0:
                print(f"Warning: Cluster {k} has no points assigned.")
                
                # Find the closest data point to this empty centroid
                distances_to_centroid = np.linalg.norm(X - self.centroids[k], axis=1)
                closest_point_idx = np.argmin(distances_to_centroid)
                
                # Manually assign the closest point to this centroid
                assigned_clusters[closest_point_idx] = k
                print(f"Assigned closest point {X[closest_point_idx]} to empty cluster {k}.")
        
        return assigned_clusters
