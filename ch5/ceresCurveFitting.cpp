#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std; 

// residual 
struct CURVE_FITTING_COST {
    CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

    // implement operator () to compute the error
    template<typename T>
    bool operator()(
        const T *const abc, // the estimated varibales, 3D vector
        T *residual) const {
            // y-exp(ax^2+bx+c)
            residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
            return true;
        }
        const double _x, _y; // x, y data
};

int main(int argc, char **argv) {
    // same as before, just making the data: 
    double ar = 1.0, br = 2.0, cr = 1.0; // ground-truth
    double ae = 2.0, be = -1.0, ce = 5.0; //intital guess   
    int N = 100; // number of data points
    double w_sigma = 1.0; 
    double inv_sigma = 1.0/w_sigma;
    cv::RNG rng;  // random number generator.

    vector<double> x_data, y_data;  // the data
    for (int i = 0; i < N; i++) {
            double x = i / 100.0;
            x_data.push_back(x);
            y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    double abc[3]{ae, be, ce};

    //construct the problem in ceres
    ceres::Problem problem;
    for (int i = 0; i < N; i++)
    {
            problem.AddResidualBlock( // add i-th residual into the problem
                // use auto-diff, template params: resudual type, output dimension, input dimesion
                // should be same as the struct written before
                new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
                    new CURVE_FITTING_COST(x_data[i], y_data[i])),
                nullptr, // kernel function, don't use here
                abc      // estimated variables
            );
    }
    // set the solver options
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary); // do optimization!
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    //get the outputs
    cout << summary.BriefReport() << endl;
    cout << "estimated abc = "; 
    for (auto a:abc)
            cout << a << " ";
    cout << endl;
    return 0; 
}
