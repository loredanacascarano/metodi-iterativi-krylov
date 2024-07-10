import numpy as np


def arnoldi_iteration(A, b, n):
    m = len(A)
    H = np.zeros((n + 1, n))
    Q = np.zeros((m, n + 1))
    Q[:, 0] = b / np.linalg.norm(b)

    for k in range(n):
        v = np.dot(A, Q[:, k])
        for j in range(k + 1):
            H[j, k] = np.dot(Q[:, j].T, v)
            v = v - H[j, k] * Q[:, j]
        H[k + 1, k] = np.linalg.norm(v)
        if H[k + 1, k] != 0 and k + 1 < m:
            Q[:, k + 1] = v / H[k + 1, k]

    return Q, H


# Example usage
A = np.array([[10, 2, 3, 0, 1],
              [2, 9, 4, 1, 0],
              [3, 4, 8, 2, 1],
              [0, 1, 2, 7, 2],
              [1, 0, 1, 2, 6]])
b = np.array([1, 0, 0, 0, 0])

Q, H = arnoldi_iteration(A, b, 2)

# Stampa le matrici in modo leggibile
print("Q:")
for row in Q:
    print(" ", ' '.join(f'{num:.1f}' for num in row))

print("H:")
for row in H:
    print(" ", ' '.join(f'{num:.1f}' for num in row))
