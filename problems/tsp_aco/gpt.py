import numpy as np

def heuristics_v2(distance_matrix):
    quadratic_reciprocal = 1 / np.power(distance_matrix, 4)
    total_distance = sum(sum(distance_matrix))
    heuristics_matrix = np.zeros((len(distance_matrix), len(distance_matrix)))

    for i in range(len(distance_matrix)):
        for j in range(i+1, len(distance_matrix)):
            heuristics_matrix[i, j] = ((total_distance ** 2) * quadratic_reciprocal[i, j])
            heuristics_matrix[j, i] = heuristics_matrix[i, j]

    return heuristics_matrix
