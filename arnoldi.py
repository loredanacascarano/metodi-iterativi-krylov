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

def solve_reduced_system(H, b, m):
    """
    Risolve il sistema ridotto H * y = beta * e1.
    """
    beta = np.linalg.norm(b)
    e1 = np.zeros(m+1)
    e1[0] = beta

    y = np.linalg.lstsq(H[:m+1, :m], e1, rcond=None)[0]
    return y

def construct_solution(V, y, m):
    """
    Costruisce la soluzione approssimata del sistema originale.
    """
    x_approx = np.dot(V[:, :m], y)
    return x_approx

# Example usage
A = np.array([[10, 2, 3, 0, 1],
              [2, 9, 4, 1, 0],
              [3, 4, 8, 2, 1],
              [0, 1, 2, 7, 2],
              [1, 0, 1, 2, 6]])
b = np.array([1, 0, 0, 0, 0])
m=4
Q, H = arnoldi_iteration(A, b, m)

# Risoluzione del sistema ridotto
y = solve_reduced_system(H, b, m)

# Costruzione della soluzione approssimata
x_approx = construct_solution(Q, y, m)

# Stampa le matrici in modo leggibile
print("Q:")
for row in Q:
    print(" ", ' '.join(f'{num:.1f}' for num in row))

print("H:")
for row in H:
    print(" ", ' '.join(f'{num:.1f}' for num in row))

# Visualizzazione della soluzione
print('Soluzione approssimata:')
print(x_approx)