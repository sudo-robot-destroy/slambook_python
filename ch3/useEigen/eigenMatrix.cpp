#include <iostream>
using namespace std; 

#include <ctime>
#include <eigen3/Eigen/Core>
// Algebraic operations of dense matrices (inverse, eigen values, etc.)
#include <eigen3/Eigen/Dense>
using namespace Eigen;

#define MATRIX_SIZE 100

// This program demos the use of basic Eigen types. 

int main(int argc, char **argv)
{
    Matrix<float, 2, 3> matrix_23; 

    // Vector3d is a Eigen::Matrix<double, 3, 1>
    Vector3d v_3d;
    Matrix<float, 3, 1> vd_3d; 

    // Matrix3d is a Eigen::Matrix<doube, 3, 3>
    Matrix3d matrix_33 = Matrix3d::Zero(); // initialized to zero

    // If you don't know the size use dynamic
    Matrix<double, Dynamic, Dynamic> matrix_dynamic; 
    // or an easier way:
    MatrixXd matrix_x; 

    // input data for initialization:
    matrix_23 << 1, 2, 3, 4, 5, 6;
    cout << "matrix 2x3 from 1 to 6: \n"
         << matrix_23 << endl; 

    // access with ()
    cout << "print matrix 2x3: " << endl;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
            cout << matrix_23(i, j) << "\t";
        cout << endl; 
    }
    // multiplication is easy:
    v_3d << 3, 2, 1;
    vd_3d << 4, 5, 6; 
    // make sure to convert explicitly (matrix_23 is floats):
    Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
    cout << "[1,2,3;4,5,6]*[3,2,1]=" << result.transpose() << endl;

    Matrix<float, 2, 1> result2 = matrix_23 * vd_3d;
    cout << "[1,2,3;4,5,6]*[4,5,6]: " << result2.transpose() << endl; 

    // Don't misjudge dimensions, see this error:
    // Eigen::Matrix<double, 2, 3> result_wrong_dim = matrix_23.cast<double>() * v_3d; 

    // some matrix operations:
    matrix_33 = Matrix3d::Random();
    cout << "random matrix: \n"
         << matrix_33 << endl;
    cout << "transpose: \n"
         << matrix_33.transpose() << endl;
    cout << "sum: " << matrix_33.sum() << endl;
    cout << "trace: " << matrix_33.trace() << endl;
    cout << "times 10: \n"
         << 10 * matrix_33 << endl;
    cout << "inverse: \n"
         << matrix_33.inverse() << endl;
    cout << "det: " << matrix_33.determinant() << endl; 

    // Eigenvalues: 
    // Real symmetric matrix is always diagonalizationable
    SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose() *
                                                  matrix_33);
    cout << "Eigen values = \n"
         << eigen_solver.eigenvalues() << endl;
    cout << "Eigen vectors = \n"
         << eigen_solver.eigenvectors() << endl; 

    // Solving equations
    // Solve matrix_NN * x = v_Nd
    // N is defined in previous macro. Direct inversion is 
    // most direct but there are a lot of operations
    Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    matrix_NN = matrix_NN * matrix_NN.transpose(); // semi-positive definite
    Matrix<double, MATRIX_SIZE, 1> v_Nd = MatrixXd::Random(MATRIX_SIZE, 1);

    clock_t time_stt = clock(); // timing
    // Direct inversion:
    Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
    cout << "time of normal inversion: "
         << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms"
         << endl;
    cout << "x = " << x.transpose() << endl << endl; 

    // Usually matrix decomposition like QR is faster:
    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout << "time of QR decomp is "
         << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms"
         << endl;
    cout << "x = " << x.transpose() << endl
         << endl; 

    // For positive definte matrices use cholesky decomp:
    time_stt = clock();
    x = matrix_NN.ldlt().solve(v_Nd);
    cout << "time of ldlt decomp is "
         << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms"
         << endl;
    cout << "x = " << x.transpose() << endl;

    return 0; 
}