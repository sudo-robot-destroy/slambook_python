#include <iostream>
#include <cmath>
using namespace std; 

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

using namespace Eigen; 
// This program demonstrates how to use the Eigen geometry module. 

int main(int argc, char **argv)
{
    // There are a variety of rotation and tranlations 
    // 3D rotation matrix using Matrix directly
    Matrix3d rotation_matrix = Matrix3d::Identity();
    // Rotation vector using angleaxis, rotate 45 deg along z:
    AngleAxisd rotation_vector(M_PI/4, Vector3d(0, 0, 1)); 
    cout.precision(3); 
    cout << "rotation matrix = \n " << rotation_vector.matrix() << endl; 

    // The matrix can also be assigned directly: 
    rotation_matrix = rotation_vector.toRotationMatrix(); 
    // coordinate transform with AngleAxis: 
    Vector3d v(1, 0, 0); 
    Vector3d v_rotated = rotation_vector * v; 
    cout << v.transpose() << " after rotation (axis angle) = " << v_rotated.transpose() << endl; 

    // or use a rotation matrix: 
    v_rotated = rotation_matrix * v; 
    cout << v.transpose() << " after rotation (matrix) = " << v_rotated.transpose() << endl;

    // You can convert matrix to euler angles
    Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0); // ZYX order (roll pitch yaw)
    cout << "yaw pitch roll = " << euler_angles.transpose() << endl;

    // Euclidean tranformation matrix using Isometry
    Isometry3d T = Isometry3d::Identity(); // it is 4x4 (even though it says 3d)
    T.rotate(rotation_vector); 
    T.pretranslate(Vector3d(1, 3, 4)); // translate by that amount
    cout << "Tranform matrix: \n" << T.matrix() << endl; 

    // Use the tranformation matrix for coordinate transformation
    Vector3d v_transformed = T * v; // Equivalent to R*v+t
    cout << "v transformed: " << v_transformed.transpose() << endl; 

    // Note there are also Affine and Projective transforms

    // Quaternions
    // AngleAxis can be assigned directly to quats and vise versa
    Quaterniond q = Quaterniond(rotation_vector); 
    cout << "quat from rot vec (xyzw): " << q.coeffs().transpose() << endl; 
    // can also assign a rotation matrix to it: 
    q = Quaterniond(rotation_matrix); 
    cout << "quat from rot mat: " << q.coeffs().transpose() << endl; 

    // Rotate a vec w/ a quat and use overloaded multiplication
    v_rotated = q * v; // note that the math is qvq^-1 
    cout << v.transpose() << " after rot (quat): " << v_rotated.transpose() << endl; 

    // Expressed by reg vec multiplicaiton it should be caluclated as follows: 
    cout << "should equal: " << (q * Quaterniond(0, 1, 0, 0) * q.inverse()).coeffs().transpose() << endl; 

    return 0; 
}