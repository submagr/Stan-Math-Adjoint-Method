#include<vector>
#include<stan/math.hpp>

template <typename F>
void numerical_jacobian (
        F dyn,
        std::vector<double> y0,
        double t0, double t1,
        std::vector<double> theta0,
        const double& eps
) {
    std::vector<double> ts({t1});
    std::vector<double> x;
    std::vector<int> x_int;
    std::vector<double> yt1 = stan::math::integrateAdj_ode_rk45<F>(dyn, y0, t0, ts, theta0, x, x_int)[0];
    std::vector<std::vector<double>> num_jacob(y0.size(), std::vector<double>(1 + 1 + theta0.size()));

    // t0
    double tl = t0 - eps/2;
    std::vector<std::vector<double>> yt1_tl = stan::math::integrateAdj_ode_rk45<F>(dyn, y0, tl, std::vector<double>({t1}), theta0, std::vector<double>(), std::vector<int>());

    double tr = t0 + eps/2;
    std::vector<std::vector<double>> yt1_tr = stan::math::integrateAdj_ode_rk45<F>(dyn, y0, tr, std::vector<double>({t1}), theta0, std::vector<double>(), std::vector<int>());

    for (int i=0; i<y0.size(); i++) {
        num_jacob[i][0] = (yt1_tr[0][i] - yt1_tl[0][i])/eps;
    }

    // t1
    double t1l = t1 - eps/2;
    std::vector<std::vector<double>> yt1_t1l = stan::math::integrateAdj_ode_rk45<F>(dyn, y0, t0, std::vector<double>({t1l}), theta0, std::vector<double>(), std::vector<int>());

    double t1r = t1 + eps/2;
    std::vector<std::vector<double>> yt1_t1r = stan::math::integrateAdj_ode_rk45<F>(dyn, y0, t0, std::vector<double>({t1r}), theta0, std::vector<double>(), std::vector<int>());

    for (int i=0; i<y0.size(); i++) {
        num_jacob[i][1] = (yt1_t1r[0][i] - yt1_t1l[0][i])/eps;
    }

    // theta
    std::vector<double> theta0Copy = theta0;
    for (int j=0; j<theta0.size(); j++) {
        double thetajl = theta0[j] - eps/2;
        theta0Copy[j] = thetajl;
        std::vector<std::vector<double>> yt1_thetajl = stan::math::integrateAdj_ode_rk45<F>(dyn, y0, t0, std::vector<double>({t1}), theta0Copy, std::vector<double>(), std::vector<int>());
        theta0Copy[j] = theta0[j];

        double thetajr = theta0[j] + eps/2;
        theta0Copy[j] = thetajr;
        std::vector<std::vector<double>> yt1_thetajr = stan::math::integrateAdj_ode_rk45<F>(dyn, y0, t0, std::vector<double>({t1}), theta0Copy, std::vector<double>(), std::vector<int>());
        theta0Copy[j] = theta0[j];

        for (int i=0; i<y0.size(); i++) {
            num_jacob[i][j+2] = (yt1_thetajr[0][i] - yt1_thetajl[0][i])/eps;
        }
    }

    for (int i=0; i<y0.size(); i++) {
        for (int j=0; j<theta0.size()+2; j++) {
            std::cout << num_jacob[i][j] << " ";
        }
        std::cout << std::endl;
    }
}
