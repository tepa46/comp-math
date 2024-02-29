#include <cmath>
#include <omp.h>
#include <functional>
#include <iostream>
#include <iomanip>

using namespace std;

const int THREADS_NUM[] = {1, 4, 8, 12};
const int NET_SIZE = 10000;
const int BLOCK_SIZE = 64;
const double EPS = 0.01;

class ApproximationNet {
public:
    vector<vector<double>> u;
    double h;

    ApproximationNet(int n, double(*f_fun)(double, double),
        double(*g_fun)(double, double)) : n(n), h(1.0 / (n + 1)) {
        f.resize(n + 2, vector<double>(n + 2, 0));
        u.resize(n + 2, vector<double>(n + 2, 0));

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
        double dmax;
        vector<double> dm = vector<double>(nb);
        do {
            dmax = 0;
            for (int nx = 0; nx < nb; nx++) {
                dm[nx] = 0;

                int i, j;
                double d;

#pragma omp parallel for shared(nx, dm) private(i, j, d)
                for (i = 0; i <= nx; i++) {
                    j = nx - i;
                    d = approximateBlock(i, j);
                    dm[i] = max(dm[i], d);
                }
            }

            for (int nx = nb - 1; nx >= 1; nx--) {
                int i, j;
                double d;

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
    vector<vector<double>> f;

    double approximateBlock(int i, int j) {
        int li = 1 + i * BLOCK_SIZE;
        int lj = 1 + j * BLOCK_SIZE;

        double dmax = 0;
        for (int ii = li; ii <= min(li + BLOCK_SIZE - 1, n); ii++) {
            for (int jj = lj; jj <= min(lj + BLOCK_SIZE - 1, n); jj++) {
                double temp = u[ii][jj];
                u[ii][jj] = 0.25 * (u[ii - 1][jj] + u[ii + 1][jj] + u[ii][jj - 1] + u[ii][jj + 1] - h * h * f[ii][jj]);
                double dm = fabs(temp - u[ii][jj]);
                dmax = max(dmax, dm);
            }
        }

        return dmax;
    }
};

// result function
double g(double x, double y) {
    return 3 * pow(x, 3) + 2 * pow(y, 5);
}

// derivative of function
double f(double x, double y) {
    return 18 * pow(x, 3) + 40 * pow(y, 2);
}

int main() {
    for (int threads_num: THREADS_NUM) {
        omp_set_num_threads(threads_num);

        ApproximationNet net = ApproximationNet(NET_SIZE, f, g);

        auto start_time = omp_get_wtime();
        net.approximate();
        auto end_time = omp_get_wtime();

        cout << "####################### " << threads_num << " ###########################\n";

        cout << "TIME: " << end_time - start_time << '\n';

        double h = net.h;
        double max_error = 0;
        for (int i = 1; i <= NET_SIZE; i++) {
            for (int j = 1; j <= NET_SIZE; j++) {
                max_error = max(max_error, fabs(g(i * h, j * h) - net.u[i][j]));
            }
        }

        cout << "MAX ERROR: " << max_error << '\n';

        long double average_error = max_error / (NET_SIZE * NET_SIZE);
        cout << "AVERAGE ERROR: " << fixed << setprecision(8) << average_error << '\n';

        cout << "#####################################################################\n";
    }
}