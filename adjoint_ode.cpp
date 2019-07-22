/*
 * Author: Shubham Agrawal
 * Adjoint Method wrapper for STAN MATH ODE Integrators
 * Based on approach: https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/adjoint.py
 * */

#include<stan/math.hpp>
#include<vector>

void flatten(
        std::vector<double>& yt1,
        std::vector<std::vector<double>>& dl_dyt1,
        std::vector<std::vector<double>>& aug_theta,
        std::vector<double>& dl_dt1,
        std::vector<double>& ret
) {
    ret.clear();
    ret.insert(ret.end(), yt1.begin(), yt1.end());
    for (auto dl_dyt1i: dl_dyt1)
        ret.insert(ret.end(), dl_dyt1i.begin(), dl_dyt1i.end());
    for (auto aug_thetai: aug_theta)
        ret.insert(ret.end(), aug_thetai.begin(), aug_thetai.end());

    ret.insert(ret.end(), dl_dt1.begin(), dl_dt1.end());
}

void inflate(
        const std::vector<double>& ret,
        double theta_size,
        double y_size,
        std::vector<double>& yt1,
        std::vector<std::vector<double>>& dl_dyt1,
        std::vector<std::vector<double>>& aug_theta,
        std::vector<double>& dl_dt1
) {
    yt1.clear();
    yt1.insert(yt1.end(), ret.begin(), ret.begin()+y_size);

    int c = y_size;

    dl_dyt1 = std::vector<std::vector<double>>(y_size, std::vector<double>(y_size));
    for (int i=0; i<y_size; i++) {
        for (int j=0; j<y_size; j++) {
            dl_dyt1[i][j] = ret[c++];
        }
    }

    aug_theta = std::vector<std::vector<double>>(y_size, std::vector<double>(theta_size));
    for (int i=0; i<y_size; i++) {
        for (int j=0; j<theta_size; j++) {
            aug_theta[i][j] = ret[c++];
        }
    }

    dl_dt1.clear();
    dl_dt1.insert(dl_dt1.end(), ret.begin()+c, ret.end());
}

namespace stan {
    namespace math {
        template <typename F>
        struct aug_dyn_ {
            F dyn;
            int theta_size, y_size, t1;

            // Remember to pass backward dynamics as F dyn.
            aug_dyn_(F dyn, int theta_size, int y_size, int t1): dyn(dyn), theta_size(theta_size), y_size(y_size), t1(t1) {};

            template<typename T0, typename T1, typename T2>
            std::vector<typename stan::return_type<T1, T2>::type>
            operator() (
                    const T0& t, const std::vector<T1>& aug_y, const std::vector<T2>& theta,
                    const std::vector<double>& x, const std::vector<int>& x_i, std::ostream* msgs
            ) const {
                std::vector<double> yt1;
                std::vector<std::vector<double>> at1;
                std::vector<std::vector<double>> dl_dtheta0;
                std::vector<double> dl_dt0;

                inflate(aug_y, theta_size, y_size, yt1, at1, dl_dtheta0, dl_dt0);

                stan::math::var tVar = t1 - t;
                std::vector<stan::math::var> yt1Var(yt1.begin(), yt1.end());
                std::vector<stan::math::var> thetaVar(theta.begin(), theta.end());
                std::vector<stan::math::var> f = dyn(tVar, yt1Var, thetaVar, std::vector<double>(), std::vector<int>(), &(std::cout));
                std::vector<double> fVal = value_of(f);


                // Calculate -a(t)
                std::vector<std::vector<double>> at1T (at1.size(), std::vector<double>(at1[0].size()));
                for (int i=0; i<at1T.size(); i++) {
                    for (int j=0; j<at1T[0].size(); j++) {
                        at1T[i][j] = -at1[i][j];
                    }
                }


                std::vector<std::vector<double>> df_dyt1(yt1.size(), std::vector<double>(yt1.size()));
                for (int i=0; i<yt1.size(); i++) {
                    f[i].grad();
                    for (int j=0; j<yt1.size(); j++) {
                        df_dyt1[i][j] = yt1Var[j].adj();
                    }
                    stan::math::set_zero_all_adjoints();
                }
                std::vector<std::vector<double>> ret_ayt1(at1T.size(), std::vector<double>(df_dyt1[0].size()));
                for (int i=0; i<ret_ayt1.size(); i++) {
                    for (int j=0; j<ret_ayt1[0].size(); j++) {
                        double sum = 0;
                        for (int k=0; k<df_dyt1.size(); k++) {
                            sum += at1T[i][k]*df_dyt1[k][j];
                        }
                        ret_ayt1[i][j] = sum;
                    }
                }

                std::vector<std::vector<double>> df_dtheta(yt1.size(), std::vector<double>(theta.size()));
                for (int i=0; i<yt1.size(); i++) {
                    f[i].grad();
                    for (int j=0; j<theta.size(); j++) {
                        df_dtheta[i][j] = thetaVar[j].adj();
                    }
                    stan::math::set_zero_all_adjoints();
                }
                std::vector<std::vector<double>> ret_atheta(at1T.size(), std::vector<double>(df_dtheta[0].size()));
                for (int i=0; i<ret_atheta.size(); i++) {
                    for (int j=0; j<ret_atheta[0].size(); j++) {
                        double sum = 0;
                        for (int k=0; k<df_dtheta.size(); k++) {
                            sum += at1T[i][k]*df_dtheta[k][j];
                        }
                        ret_atheta[i][j] = sum;
                    }
                }

                std::vector<double> df_dt(yt1.size());
                for (int i=0; i<yt1.size(); i++){
                    f[i].grad();
                    df_dt[i] = tVar.adj();
                    stan::math::set_zero_all_adjoints();
                }
                std::vector<double> ret_at(df_dt.size());
                for (int i=0; i<at1T.size(); i++) {
                    double sum = 0;
                    for (int k=0; k<df_dt.size(); k++) {
                        sum += at1T[i][k]*df_dt[k];
                    }
                    ret_at[i] = sum;
                }

                // flatten aug_y0 into a vector
                std::vector<typename stan::return_type<T1, T2>::type> ret_aug_y;
                flatten(fVal, ret_ayt1, ret_atheta, ret_at, ret_aug_y);

                return ret_aug_y;
            }
        };

        template<typename F>
        struct dyn_back_{
            F dyn;
            dyn_back_(F dyn): dyn(dyn) {};

            template<typename T0, typename T1, typename T2>
            std::vector<typename stan::return_type<T1, T2>::type>
            operator() (
                    const T0& t, const std::vector<T1>& y, const std::vector<T2>& theta,
                    const std::vector<double>& x, const std::vector<int>& x_i, std::ostream* msgs
            ) const {
                std::vector<typename stan::return_type<T1, T2>::type> dydt = dyn(t, y, theta, x, x_i, msgs);
                for (int i=0; i<dydt.size(); i++) {
                    dydt[i] *= -1;
                }
                return dydt;
            }
        };

        template<typename F>
        std::vector<std::vector<double>> integrateAdj_ode_rk45(
                const F& dyn, const std::vector<double>& y0, const double& t0,
                const std::vector<double>& ts,
                const std::vector<double>& theta0,
                const std::vector<double>& x, const std::vector<int>& x_int,
                std::ostream* msgs = nullptr
        ) {
            return integrate_ode_rk45(dyn, y0, t0, ts, theta0, x, x_int, msgs);
        }

        template<typename F>
        std::vector<std::vector<var>> integrateAdj_ode_rk45(
                const F& dyn, const std::vector<var>& y0, const var& t0,
                const std::vector<var>& ts_in,
                const std::vector<var>& theta0,
                const std::vector<double>& x, const std::vector<int>& x_int,
                std::ostream* msgs = nullptr
        ) {
            var t1 = ts_in[0];

            std::vector<double> y0Val = value_of(y0);
            std::vector<double> theta0Val = value_of(theta0);
            double t0Val = value_of(t0);
            double t1Val = value_of(t1);

            const dyn_back_<F> dyn_back(dyn);
            const aug_dyn_<dyn_back_<F>> aug_dyn(dyn_back, theta0.size(), y0.size(), t1Val);


            // y(t1)
            std::vector<std::vector<double>> yts1 = integrateAdj_ode_rk45(
                    dyn, y0Val, t0Val, std::vector<double>({t1Val}),
                    theta0Val, x, x_int, msgs
            );
            std::vector<double> yt1 = yts1[0];

            // dL/dy(t1):: L here is same as y(t1) - Hence, identity Jacobian.
            std::vector<std::vector<double>> dl_dyt1(y0.size(), std::vector<double>(y0.size(), 0));
            for (int i=0; i<dl_dyt1.size(); i++) {
                for (int j=0; j<dl_dyt1[0].size(); j++) {
                    if (i == j) {
                        dl_dyt1[i][j] = 1;
                    }
                }
            }

            // aug_theta
            std::vector<std::vector<double>> aug_theta(y0.size(), std::vector<double>(theta0.size(), 0));

            // -dL/dt1 = -Transpose(dL/dy(t1)) * dyn(y(t1), t1, theta)
            std::vector<double> dl_dt1 = dyn(t1Val, yt1, theta0Val, std::vector<double>(), std::vector<int>(), &(std::cout));
            for (int i=0; i<dl_dt1.size(); i++) {
                dl_dt1[i] *= -1;
            }

            std::vector<double> aug_y0;
            flatten(yt1, dl_dyt1, aug_theta, dl_dt1, aug_y0);
            std::vector<double> ts = {t1Val - t0Val};

            std::vector<std::vector<double>> aug_y_ = integrate_ode_rk45(
                    aug_dyn,
                    aug_y0,
                    0,
                    ts,
                    theta0Val,
                    x,
                    x_int,
                    msgs
            );

            std::vector<double> aug_y = aug_y_[0];

            std::vector<double> yt1_temp;
            std::vector<std::vector<double>> dl_dyt0;
            std::vector<std::vector<double>> dl_dtheta0;
            std::vector<double> dl_dt0;

            inflate(
                    aug_y,
                    theta0.size(),
                    y0.size(),
                    yt1_temp,
                    dl_dyt0,
                    dl_dtheta0,
                    dl_dt0
            );

            for (int i=0; i<dl_dt1.size(); i++)
                dl_dt1[i] *= -1;

            std::vector<var> params;
            params.insert(params.end(), y0.begin(), y0.end());
            params.insert(params.end(), theta0.begin(), theta0.end());
            params.insert(params.end(), t0);
            params.insert(params.end(), t1);

            std::vector<var> yt1Var(yt1.size());
            for (int i=0; i<yt1Var.size(); i++) {
                std::vector<double> val_jacob_i;
                val_jacob_i.insert(val_jacob_i.end(), dl_dyt0[i].begin(), dl_dyt0[i].end());
                val_jacob_i.insert(val_jacob_i.end(), dl_dtheta0[i].begin(), dl_dtheta0[i].end());
                val_jacob_i.insert(val_jacob_i.end(), dl_dt0[i]);
                val_jacob_i.insert(val_jacob_i.end(), dl_dt1[i]);
                yt1Var[i] = precomputed_gradients(yt1[i], params, val_jacob_i);
            }
            return std::vector<std::vector<var>>({yt1Var});
        }
    }
}
