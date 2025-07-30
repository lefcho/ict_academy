import numpy as np


def normalize_matrix(matrix):
    column_norms = np.sqrt((matrix**2).sum(axis=0))
    return matrix / column_norms


def calc_distance(matrix, v):
    return np.sqrt(((matrix - v)**2).sum(axis=1))


def topsis(matrix):
    weights = np.array([0.1, 0.4, 0.3, 0.2], dtype=float)

    normalize_matrix(matrix)
    matrix *= weights
    matrix[:, -1] *= -1

    print(matrix)

    ideal = matrix.max(axis=0)
    antiideal = matrix.min(axis=0)

    ideal_alt = calc_distance(matrix, ideal)
    antiideal_alt = calc_distance(matrix, antiideal)

    closeness = antiideal_alt / (ideal_alt + antiideal_alt)

    return closeness


matrix = np.array([[7, 9, 9, 8],
                    [8, 7, 8, 7],
                    [9, 6, 8, 9],
                    [6, 7, 8, 6]], dtype=float)

closeness = topsis(matrix)

print(np.sort(closeness))
