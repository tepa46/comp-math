import numpy as np
from math import sqrt, sin, cos
import matplotlib

matplotlib.use('TkAgg')  # Or 'Qt5Agg', 'GTK3Agg' etc., depending on your preference
import matplotlib.pyplot as plt


def phi_j(j, xj, x_nodes, N):
    h = x_nodes[1] - x_nodes[0]

    if j == 0:
        return (x_nodes[1] - xj) / h if (x_nodes[0] <= xj) and (xj <= x_nodes[1]) else 0
    elif j == N:
        return (xj - x_nodes[N - 1]) / h if (x_nodes[N] <= xj) and (x_nodes[N - 1] >= xj) else 0
    else:
        if (x_nodes[j - 1] <= xj) and (x_nodes[j] >= xj):
            return (xj - x_nodes[j - 1]) / h
        elif (x_nodes[j] <= xj) and (x_nodes[j + 1] >= xj):
            return (x_nodes[j + 1] - xj) / h
        else:
            return 0


# https://en.wikibooks.org/wiki/Algorithm_Implementation/Linear_Algebra/Tridiagonal_matrix_algorithm
def tdma(a, b, c, d):
    """
      Решает систему уравнений с трехдиагональной матрицей методом прогона Томаса.

      Args:
        a: Массив с элементами под главной диагональю.
        b: Массив с элементами на главной диагонали.
        c: Массив с элементами над главной диагональю.
        d: Вектор правой части.

      Returns:
        Вектор решения системы уравнений.
      """
    n = len(b)

    for i in range(1, n):
        m = a[i] / b[i - 1]
        b[i] = b[i] - m * c[i - 1]
        d[i] = d[i] - m * d[i - 1]

    y = np.zeros(n)
    y[n - 1] = d[n - 1] / b[n - 1]

    for i in range(n - 2, -1, -1):
        y[i] = (d[i] - c[i] * y[i + 1]) / b[i]

    return y


def process_line(lambada, j, x, N):
    h = x[1] - x[0]

    if j <= 1:
        a = None
    else:
        a = (-1.0 / 6.0) * (-6 + lambada * (x[j - 1] - x[j]) ** 2) * (x[j - 1] - x[j]) / (h ** 2)

    if j >= N - 1:
        c = None
    else:
        c = (-1.0 / 6.0) * (-6 + lambada * (x[j] - x[j + 1]) ** 2) * (x[j] - x[j + 1]) / (h ** 2)

    b = (x[j + 1] + lambada * x[j + 1] * (x[j] ** 2 - x[j] * x[j + 1] + (x[j + 1] ** 2) / 3) - x[j - 1] - lambada * x[
        j - 1] * (x[j] ** 2 - x[j] * x[j - 1] + (x[j - 1] ** 2) / 3)) / (h ** 2)

    lambada_sqrt = sqrt(lambada)
    d = (2 * (-(x[j] - x[j + 1]) * lambada_sqrt * cos(x[j] * lambada_sqrt) + sin(x[j] * lambada_sqrt) - sin(
        x[j + 1] * lambada_sqrt)) + 2 * (
                 -lambada_sqrt * (x[j] - x[j - 1]) * cos(x[j] * lambada_sqrt) + sin(x[j] * lambada_sqrt) - sin(
             lambada_sqrt * x[j - 1]))) / h

    return a, b, c, d


def build_equations(lambada, x, N):
    a = np.zeros(N - 1)
    b = np.zeros(N - 1)
    c = np.zeros(N - 1)
    d = np.zeros(N - 1)

    for j in range(1, N):
        a[j - 1], b[j - 1], c[j - 1], d[j - 1] = process_line(lambada, j, x, N)

    return a, b, c, d


def f_real_answer(x, _lambda):
    return np.sin(np.sqrt(_lambda) * x)


def get_right_grid_index(x_grid, x, n):
    l, r = 0, n

    while r - l > 1:
        middle = (l + r) // 2
        if x > x_grid[middle]:
            l = middle
        else:
            r = middle
    return l, r


def approxi_res(x, x_grid, _lambda, n, y):
    l, r = get_right_grid_index(x_grid, x, n)
    y_1 = y[l]
    y_2 = y[r]
    phi_1 = phi_j(l, x, x_grid, n)
    phi_2 = phi_j(r, x, x_grid, n)

    return y_1 * phi_1 + y_2 * phi_2


def calc_error(n, x_grid, loh, _lambda, y):
    max_error = 0
    nn = n * 10
    nh = loh / nn
    for i in range(nn):
        x = i * nh
        approxi_val = approxi_res(x, x_grid, _lambda, n, y)
        real_val = f_real_answer(x, _lambda)
        max_error = max(max_error, abs(real_val - approxi_val))
    return max_error


def draw_graph(h, error):
    plt.title("Max error has the same order as h^2")
    plt.loglog(h, error, 'o', color='red')
    # plt.scatter(h, error, color='red', marker='o', s=50)
    plt.xlabel('h')
    plt.ylabel('error')
    plt.grid(True)
    plt.savefig("error_graph.png")


def main():
    lambdas = [1 / 16, 1 / 128, 1 / 256]
    grid_sizes = [10, 100, 1000, 10000, 100000]

    hs = []
    errors = []

    for lambada in lambdas:
        for N in grid_sizes:
            loh = 4 * np.pi / sqrt(lambada)
            A = 0.0
            B = np.pi * loh
            x = np.linspace(A, B, N + 1)
            h = x[1] - x[0]

            a, b, c, d = build_equations(lambada, x, N)

            y_h = tdma(a, b, c, d)

            error = calc_error(N, x, loh, lambada, y_h)
            hs.append(h)
            errors.append(error)
            print(error)

    draw_graph(hs, errors)


if __name__ == '__main__':
    main()
