#include <cmath>
#include <functional>

using namespace std;

const int THREADS_NUM[] = {1, 4, 8, 12};
const int NET_SIZE = 10000;
const int BLOCK_SIZE = 64;
const double EPS = 1e-1;

class ApproximationNet {
public:
    vector<vector<long double>> u;

    ApproximationNet(int n, long double(*f_fun)(long double, long double),
                     long double(*g_fun)(long double, long double)) : n(n), h(1.0 / (n + 1)) {
        f.resize(n + 2, vector<long double>(n + 2, 0));
        u.resize(n + 2, vector<long double>(n + 2, 0));

        // derivatives
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                f[i][j] = f_fun(i * h, j * h);
            }
        }

        // boundary points
        for (int i = 0; i <= n + 1; i++) {
            u[i][0] = g_fun(i * h, 0);
            u[i][n + 1] = g_fun(i * h, (n + 1) * h);
            u[0][i] = g_fun(0, i * h);
            u[n + 1][i] = g_fun((n + 1) * h, i * h);
        }
    }

    void approximate() {
        int nb = n / BLOCK_SIZE + (n % BLOCK_SIZE != 0 ? 1 : 0);
        long double dmax;
        vector<long double> dm = vector<long double>(nb);
        do {
            dmax = 0;
            for (int nx = 0; nx < nb; nx++) {
                dm[nx] = 0;

                int i, j;
                long double d;

#pragma omp parallel for shared(nx, dm) private(i, j, d)
                for (i = 0; i <= nx; i++) {
                    j = nx - i;
                    d = approximateBlock(i, j);
                    dm[i] = max(dm[i], d);
                }
            }

            for (int nx = nb - 1; nx >= 1; nx--) {
                int i, j;
                long double d;

#pragma omp parallel for shared(nx, dm) private(i, j, d)
                for (i = nb - nx; i < nb; i++) {
                    j = 2 * (nb - 1) - nx - i + 1;
                    d = approximateBlock(i, j);
                    dm[i] = max(dm[i], d);
                }
            }

            for (int i = 0; i < nb; i++) {
                dmax = max(dmax, dm[i]);
            }
        } while (dmax > EPS);
    }

private:
    int n;
    double h;
    vector<vector<long double>> f;

    long double approximateBlock(int i, int j) {
        int li = 1 + i * BLOCK_SIZE;
        int lj = 1 + j * BLOCK_SIZE;

        long double dmax = 0;
        for (int ii = li; ii <= min(li + BLOCK_SIZE - 1, n); ii++) {
            for (int jj = lj; jj <= min(lj + BLOCK_SIZE - 1, n); jj++) {
                long double temp = u[ii][jj];
                u[ii][jj] = 0.25 * (u[ii - 1][jj] + u[ii + 1][jj] + u[ii][jj - 1] + u[ii][jj + 1] - h * h * f[ii][jj]);
                long double dm = fabs(temp - u[ii][jj]);
                dmax = max(dmax, dm);
            }
        }

        return dmax;
    }
};

// result function
long double g(long double x, long double y) {
    return 52;
}

// derivative of the function
long double f(long double x, long double y) {
    return 0;
}
