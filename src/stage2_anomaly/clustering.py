"""Gate 3: leader-follower incremental clustering, ported near-verbatim from
cloud_service.gate3_assign so offline evaluation exercises the same logic
the live deployment runs."""

import numpy as np


class LeaderFollowerClusterer:
    def __init__(self, cluster_radius: float, min_cluster_size: int = 30):
        self.cluster_radius = cluster_radius
        self.min_cluster_size = min_cluster_size
        self.cluster_centroids: list = []
        self.cluster_counts: list = []
        self.promoted_clusters: set = set()

    def assign(self, z: np.ndarray):
        if not self.cluster_centroids:
            self.cluster_centroids.append(z.copy())
            self.cluster_counts.append(1)
            return "Zero-day-Unclustered", 0

        dists = [np.linalg.norm(z - c) for c in self.cluster_centroids]
        best = int(np.argmin(dists))

        if dists[best] <= self.cluster_radius:
            n = self.cluster_counts[best]
            self.cluster_centroids[best] = (self.cluster_centroids[best] * n + z) / (n + 1)
            self.cluster_counts[best] += 1
            idx = best
        else:
            self.cluster_centroids.append(z.copy())
            self.cluster_counts.append(1)
            idx = len(self.cluster_centroids) - 1

        if self.cluster_counts[idx] >= self.min_cluster_size:
            self.promoted_clusters.add(idx)

        if idx in self.promoted_clusters:
            return f"Candidate-Class-{idx+1}", idx
        return "Zero-day-Unclustered", idx
