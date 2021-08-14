import numpy as np

from base import BaseEstimator


class KMeans(BaseEstimator):
    def __init__(self, k, max_iterations):
        self.k = k
        self.max_iterations = max_iterations

        self._clusters = None
        self._centroids = None
        self._labels = None

    def fit(self, X):
        self._centroids = self._init_centroids(X)

        for iterations in range(self.max_iterations):
            # assign clusters
            self._clusters = self._assign_cluster(X)

            old_centroids = self._centroids
            # calculate new centroids
            self._centroids = self._calculate_centroids(X)

            if self._is_converged(old_centroids, self._centroids):
                break

        self._labels = self._get_labels(X, self._clusters)

    def predict(self, X):
        clusters = self._assign_cluster(X)
        labels = self._get_labels(X, clusters)
        return labels

    def _init_centroids(self, X):
        (n_samples, _) = X.shape
        random_idxs = np.random.choice(n_samples, self.k, replace=False)
        centroids = [X[i] for i in random_idxs]
        return np.array(centroids)

    def _assign_cluster(self, X):
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(X):
            closest_centroid_idx = np.argmin(np.sqrt(np.sum((sample - self._centroids) ** 2, axis=1)))
            clusters[closest_centroid_idx].append(idx)
        return clusters

    def _calculate_centroids(self, X):
        centroids = []
        for cluster_idxs in self._clusters:
            centroid = np.mean(X[cluster_idxs], axis=0)
            centroids.append(centroid)
        return np.array(centroids)

    def _is_converged(self, old_centroids, new_centroids):
        return np.sum(old_centroids - new_centroids) == 0

    def _get_labels(self, X, clusters):
        labels = np.empty(X.shape[0])
        for cluster_idx, cluster in enumerate(clusters):
            labels[cluster] = cluster_idx
        return labels
