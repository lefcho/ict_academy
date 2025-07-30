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
            return result, next_vec, vector

        m = m_next



matrix = np.array([[1, 0.5, 3],
                    [2, 1, 4],
                    [0.333, 0.25, 1]], dtype=float)

res, final_vec, prev_vec = weight_calc(matrix, threshold=0.01)

print("prev vector:", prev_vec)
print("final vector:", final_vec)
print("diff:", res)
