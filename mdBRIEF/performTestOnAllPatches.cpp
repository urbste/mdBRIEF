/**
 * @file norm.cpp
 * @brief mex interface for cv::norm
 * @ingroup core
 * @author Amro
 * @date 2015
 */
#include "mexopencv.hpp"
using namespace std;
using namespace cv;
#include <omp.h>
/**
 * Main entry called from Matlab
 * @param nlhs number of left-hand-side arguments
 * @param plhs pointers to mxArrays in the left-hand-side
 * @param nrhs number of right-hand-side arguments
 * @param prhs pointers to mxArrays in the right-hand-side
 */
inline void rotate_test(float angle, const Point& p1,const Point& p2, Point& p1o, Point& p2o)
{
    float cosa = cos(angle);
    float sina = sin(angle);
    
    int x1 = p1.x-16; // -1 for matlab to c++ and -15 for patchmean
    int y1 = p1.y-16;
    int x2 = p2.x-16;
    int y2 = p2.y-16;
    
    int rotx1 = cvRound(cosa*x1-sina*y1+16);
    int roty1 = cvRound(sina*x1+cosa*y1+16);
    int rotx2 = cvRound(cosa*x2-sina*y2+16);
    int roty2 = cvRound(sina*x2+cosa*y2+16);    
    
    if (rotx1 < 0)
        rotx1 = 0;
    if (rotx1 > 31)
        rotx1 = 31;

    if (roty1 < 0)
        roty1 = 0;
    if (roty1 > 31)
        roty1 = 31; 

    if (rotx2 < 0)
        rotx2 = 0;
    if (rotx2 > 31)
        rotx2 = 31;

    if (roty2 < 0)
        roty2 = 0;
    if (roty2 > 31)
        roty2 = 31;
    
    p1o.y = rotx1;
    p1o.x = roty1;
    p2o.y = rotx2;
    p2o.x = roty2;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    
    // Argument vector
    vector<MxArray> rhs(prhs, prhs+nrhs);
    // all patches N x (patch_size^2)
    Mat patches(rhs[0].toMat(CV_8UC1));
    // all tests 4 x nr_tests
    Mat tests(rhs[1].toMat(CV_32SC1));
    // all tests 4 x nr_tests
    Mat rotations(rhs[2].toMat(CV_32FC1));
    // result
    int nrPatches = patches.rows;
    int nrTests = tests.cols;
    Mat means = Mat::zeros(3,nrTests, CV_32FC1);

#pragma omp parallel for num_threads(2)
    for (int t = 0; t < nrTests; ++t)
    {
        Mat bitstring = Mat::zeros(nrPatches,1,CV_8UC1);
//         Point p1(tests.at<int>(1,t)-1,
//                  tests.at<int>(0,t)-1);
//         //mexPrintf("hier3");
//         Point p2(tests.at<int>(3,t)-1,
//                  tests.at<int>(2,t)-1);
        
        for (int i=0; i < nrPatches; ++i)
        {
            float angle = rotations.ptr<float>(0)[i];
            Point p1(tests.ptr<int>(0)[t],tests.ptr<int>(1)[t]);
            Point p2(tests.ptr<int>(2)[t],tests.ptr<int>(3)[t]);
            Point p1o,p2o;
            rotate_test(angle, p1,p2, p1o,p2o);
            Mat patch = patches.row(i).reshape(1,32);
            // perform test
            bitstring.ptr<uchar>(i)[0] = (uchar)(patch.ptr<uchar>(p1o.y)[p1o.x] < 
                                                 patch.ptr<uchar>(p2o.y)[p2o.x]);
        }
        float p = (float)cv::mean(bitstring)[0];
        means.ptr<float>(0)[t] = p*(1-p); // variance
        means.ptr<float>(1)[t] = t+1;     // index
        means.ptr<float>(2)[t] = p;       // mean
    }
//     
//     char out[100];
//     sprintf(out,"Finished variance calculation...\n");
//     mexPrintf(out);
    
    plhs[0] = MxArray(means);
}
