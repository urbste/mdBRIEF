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
    
    int x1 = p1.x-16;
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
    // rotations
    Mat rotations(rhs[1].toMat(CV_32FC1));
    // all sorted means
    Mat sorted_means_indices(rhs[2].toMat(CV_32SC1));
    // all ests
    Mat all_tests(rhs[3].toMat(CV_32SC1));
    // descriptor size
    int desc_size = rhs[4].toInt();
     // correlation thresh
    float thresh = rhs[5].toFloat(); 
       
    // result
    int nrPatches = patches.rows;
    int nrTests = all_tests.cols;
    Mat global_best_tests = Mat::zeros(4,desc_size, CV_32SC1);
    Mat idxAndDist = Mat::zeros(2,desc_size, CV_32FC1);
    // set the first test which has the highest variance
    int firstIdx = sorted_means_indices.at<int>(0,0)-1;
    all_tests.col(firstIdx).copyTo(global_best_tests.col(0));
    
    idxAndDist.ptr<float>(0)[0] = firstIdx;
    idxAndDist.ptr<float>(1)[0] = 0.0;
    
    int nr_global_best_tests = 1;
    while (nr_global_best_tests < desc_size)
    {     
        char out[100];  
        sprintf(out,"current thresh: %f\n",thresh);
        mexPrintf(out);
        mexEvalString("drawnow"); // to dump string.
        mexEvalString("pause(.001);"); // to dump string.
        for (int c = 1; c < sorted_means_indices.cols; ++c)
        {
            // get index of next test
            int testIdx = sorted_means_indices.ptr<int>(0)[c]-1;
            // from all tests
            // the tests are inverse: 0,1 -> 1,0 otherwise we had to 
            // transpose the patch which is not O(1)
            Point p1(all_tests.ptr<int>(0)[testIdx]-1,
                     all_tests.ptr<int>(1)[testIdx]-1);
            Point p2(all_tests.ptr<int>(2)[testIdx]-1,
                     all_tests.ptr<int>(3)[testIdx]-1);
            
            bool good = 1;
            float bestDist = 0.0f;
            float bestIdx = 0.0f;
            for (int r = 0; r < nr_global_best_tests; ++r)
            {
                bool bit1 = 0;
                bool bit2 = 0;
                float dist = 0;
                // from all saved global best tests
                Point p3(global_best_tests.ptr<int>(0)[r]-1,
                         global_best_tests.ptr<int>(1)[r]-1);
                Point p4(global_best_tests.ptr<int>(2)[r]-1,
                         global_best_tests.ptr<int>(3)[r]-1);
                     
                for (int p = 0; p < nrPatches; ++p)
                {
                    float angle = rotations.at<float>(p);
                    Point p1o,p2o,p3o,p4o;
                    // perform tests on patch p
                    rotate_test(angle, p1,p2,p1o,p2o); // we switch x and y in here
                    rotate_test(angle, p3,p4,p3o,p4o); // we switch x and y in here   
                    Mat patch = patches.row(p).reshape(0,32);
                    bit1 = (patch.ptr<uchar>(p1o.y)[p1o.x] < patch.ptr<uchar>(p2o.y)[p2o.x]);
                    bit2 = (patch.ptr<uchar>(p3o.y)[p3o.x] < patch.ptr<uchar>(p4o.y)[p4o.x]);
                    //dist = dist + (float)(bit1^bit2);
                    dist += (float)abs(bit1-bit2);
                }
                // now lets see how correlated the test is
                dist = abs(2.f/nrPatches * dist - 1.0);
                // if the distance is bigger with respect to any test break
                // and get a new test
                if (dist > thresh)
                {
                    good = 0;
                    break;    
                }
                else bestDist = dist;
            }
            // if the correlation is low, add the test
            if (good)
            {
                // save some statistics
                idxAndDist.ptr<float>(0)[nr_global_best_tests] = static_cast<float>(testIdx);
                idxAndDist.ptr<float>(1)[nr_global_best_tests] = bestDist;
                all_tests.col(testIdx).copyTo(global_best_tests.col(nr_global_best_tests));
                char out[100];  
                sprintf(out,"Added test nr %d, idx %d, dist %f\n",nr_global_best_tests, (int)testIdx, bestDist);
                mexPrintf(out);
                mexEvalString("drawnow"); // to dump string. 
                mexEvalString("pause(.001);"); // to dump string.
                ++nr_global_best_tests;
                if (nr_global_best_tests == desc_size)
                    break;  
            }
        }
        thresh = thresh + 0.1f;
    }
    plhs[0] = MxArray(global_best_tests);
    plhs[1] = MxArray(idxAndDist);
}
