import numpy as np


def normalize_row(matrix):
    row_sums = np.sum(matrix, axis=1)
    total_sum = np.sum(matrix)
    return row_sums / total_sum


def weight_calc(m, threshold):

    result = float('inf')

    while True:
        m_next = m.dot(m)

        vector = normalize_row(m)
        next_vec = normalize_row(m_next)

        result = np.max(np.abs(vector - next_vec))

        if result < threshold:
            return next_vec

        m = m_next


def normalize_matrix(matrix):


    column_norms = np.sqrt((matrix**2).sum(axis=0))
    return matrix / column_norms


def calc_distance(matrix, v):
    return np.sqrt(((matrix - v)**2).sum(axis=1))


def topsis(matrix, criteria_cmp, threshold=0.01):

    weights = weight_calc(criteria_cmp, threshold)
    print(weights)
    normalize_matrix(matrix)
    matrix *= weights
    matrix *= np.array([1, 1, 1, -1], dtype=float)

    ideal = matrix.max(axis=0)
    antiideal = matrix.min(axis=0)

    ideal_dist = calc_distance(matrix, ideal)
    antiideal_dist = calc_distance(matrix, antiideal)

    closeness = antiideal_dist / (ideal_dist + antiideal_dist)

    return closeness


matrix = np.array([[7, 9, 9, 8],
                    [8, 7, 8, 7],
                    [9, 6, 8, 9],
                    [6, 7, 8, 6]], dtype=float)


criteria_array = np.array([
    [1,   3,   4,   2],
    [1/3, 1,   2,   1/3],
    [1/4, 1/2, 1,   1/5],
    [1/2, 3,   5,   1]
], dtype=float)


closeness = topsis(matrix, criteria_array, threshold=0.01)


print(np.round(np.sort(closeness), 3))
