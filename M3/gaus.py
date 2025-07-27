import numpy as np

# matrix = np.array([[-3, -1, 2, -11], [2, 1, -1, 8], [-2, 1, 2, -3]], dtype=float)

# matrix[0] = matrix[0] / matrix[0][0]

# matrix[1] = matrix[1] - matrix[0] * matrix[1][0]

# matrix[2] = matrix[2] - matrix[0] * matrix[2][0]

# matrix[1] = matrix[1] / matrix[1][1]

# matrix[2] = matrix[2] - matrix[1] * matrix[2][1]

# matrix[2] = matrix[2] / matrix[2][2]


matrix = np.array([[-3, -1, 2, -11], [2, 1, -1, 8], [-2, 1, 2, -3]], dtype=float)

rows = len(matrix)

for i in range(rows):
    
    if matrix[i][i] == 0:
        for c in range(i + 1, rows):
            if matrix[c][i] != 0:
                matrix[[i, c]] = matrix[[c, i]]
                break

    matrix[i] /= matrix[i][i]

    for j in range(i + 1, rows):
        matrix[j] -= matrix[i] * matrix[j][i]

for i in range(rows - 1, -1, -1):
    matrix[i] /= matrix[i][i]

    for j in range(i - 1, -1, -1):
        matrix[j] -= matrix[i] * matrix[j][i]

print(matrix)
