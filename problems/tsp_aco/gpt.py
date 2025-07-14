import numpy as np
from scipy.stats import zscore

def heuristics_v2(distance_matrix: np.ndarray, competitive_decay_rate: float = 0.95, z_score_threshold: float = 3, mad_factor: float = 1.4826, gamma: float = 1.0) -> np.ndarray:
    """
    An improved TSP heuristics function that incorporates MAD scaling, Z-score normalization, and competitive edge prioritization for efficient optimization.
    Customize the competitive_decay_rate and z_score_threshold parameters for balance between nearby cities, outliers, and competitive edges.
    """

    means = np.mean(distance_matrix, axis=0)
    mad = np.median(np.abs(distance_matrix - means)) * mad_factor
    scaled_distances_mad = (distance_matrix - means) / (mad)

    scaled_distances_zscore = zscore(distance_matrix)

    competitive_factors = np.exp(-competitive_decay_rate * scaled_distances_zscore)

    sparsified_distances = np.where((scaled_distances_mad > z_score_threshold) | (competitive_factors < 0.1), 0, scaled_distances_mad)

    min_scaled_distance = np.min(sparsified_distances, axis=0)
    max_scaled_distance = np.max(sparsified_distances, axis=0)

    normalized_distances = (sparsified_distances - min_scaled_distance) / (max_scaled_distance - min_scaled_distance)
    normalized_distances = 1 / (normalized_distances + 1e-8)

    return normalized_distances
