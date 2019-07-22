#include<adjoint_ode.cpp>
#include<numerical_jacobian.cpp>
#include<vector>
#include<iostream>


struct cartpole_dyn {
    template <typename T0, typename T1, typename T2>
    std::vector<typename stan::return_type<T1, T2>::type>
    operator()(const T0 &t, const std::vector<T1> &y,
               const std::vector<T2> &theta, const std::vector<double> &x,
               const std::vector<int> &x_i, std::ostream *msgs) const {
        std::vector<typename stan::return_type<T1, T2>::type> dydt(3);
        dydt[0] = (1.0/(theta[3]*theta[1]*cos(y[1]) - theta[1]*theta[2] - theta[1]*theta[3]))*(-1*theta[3]*theta[1]*theta[0]*sin(y[1])*cos(y[1]) - theta[1]*theta[4] - theta[3]*theta[1]*theta[1]*y[2]*y[2]*sin(y[1]));
        dydt[1] = y[2];
        dydt[2] = (1.0/(theta[3]*theta[1]*cos(y[1]) - theta[1]*theta[2] - theta[1]*theta[3]))*((theta[2]+theta[3])*theta[0]*sin(y[1]) + theta[4]*cos(y[1]) + theta[3]*theta[1]*y[2]*y[2]*sin(y[1])*cos(y[1]));
        return dydt;
    }
};


int main() {
    std::vector<stan::math::var> y0 = {
        0,
        stan::math::pi()/4,
        0
    };
    std::vector<stan::math::var> theta0 = {
        9.8, // g
        1, // l
        1, // m1
        1, // m2
        10 // F
    };
    stan::math::var t0 = 0;
    std::vector<stan::math::var> ts = {0.3};
    cartpole_dyn dyn;

    std::vector<std::vector<stan::math::var>> ys = stan::math::integrateAdj_ode_rk45<cartpole_dyn>(
            dyn, y0, t0, ts, theta0,
            std::vector<double>({}), std::vector<int>({})
    );
    std::vector<stan::math::var> y1 = ys[0];
    stan::math::var t1 = ts[0];

    std::cout << "y(t1): ";
    for (int i=0; i<y1.size(); i++) {
        std::cout << y1[i] << " ";
    }
    std::cout << std::endl << std::endl;

    std::cout << "\t Analytical Gradient Transpose" << std::endl;
    for (int i=0; i<y1.size(); i++) {
        y1[i].grad();
        std::cout << t0.adj() << " " << t1.adj() << " ";
        for (int j=0; j<theta0.size(); j++) {
            std::cout << theta0[j].adj() << " ";
        }
        std::cout << std::endl;
        stan::math::set_zero_all_adjoints();
    }
    std::cout << std::endl << std::endl;

    std::cout << "\t Numerical Gradient Transpose" << std::endl;
    numerical_jacobian<cartpole_dyn>(
            dyn, value_of(y0), t0.val(), t1.val(),
            value_of(theta0), 0.001
    );
    std::cout << std::endl;

    return 0;
}
