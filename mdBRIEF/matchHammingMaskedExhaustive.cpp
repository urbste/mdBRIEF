/**
 * @file matchHamming_binboost.cpp
 * @brief mex interface for BinBoost
 * @author Steffen Urban
 * @date 2015
 */
#include "mexopencv.hpp"

using namespace std;
using namespace cv;

int matchHammingMasked(const unsigned __int64* descr_i,
                    const unsigned __int64* descr_j,
                    const unsigned __int64* mi,
                    const unsigned __int64* mj,
                    const Mat& mask1,
                    const Mat& mask2,
                    int dim)
{
    unsigned __int64 distL = 0, distR = 0;
    double nL = (double)cv::sum(mask1)[0];
    double nR = (double)cv::sum(mask2)[0];
    
    for (int i = 0; i < dim; ++i) 
    {
        unsigned __int64 axorb = descr_i[i] ^ descr_j[i];
        unsigned __int64 xormaskedL = axorb & mi[i];
        unsigned __int64 xormaskedR = axorb & mj[i];
        distL += __popcnt64(xormaskedL);
        distR += __popcnt64(xormaskedR);
    }
    

    double n = nL + nR;
    double wL = nL / n;
    double wR = nR / n;
    int res = (int)(distL*wL + distR*wR);
    
    //     for (int i = 0; i < dim; ++i) 
//     {
//         unsigned __int64 axorb = descr_i[i] ^ descr_j[i];
//         unsigned __int64 xormaskedL = axorb & mi[i];
//         unsigned __int64 xormaskedR = axorb & mj[i];
//         distL += __popcnt64(xormaskedL);
//         distL += __popcnt64(xormaskedR);
//     }
    //int res = static_cast<int>(distL/2);
    return res;
}


void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
    // Argument vector
    vector<MxArray> rhs(prhs,prhs+nrhs);

    Mat desc1 = rhs[0].toMat(CV_8UC1);
    Mat desc2 = rhs[1].toMat(CV_8UC1);
    Mat mask1 = rhs[2].toMat(CV_8UC1);
    Mat mask2 = rhs[3].toMat(CV_8UC1);
    int thresh = rhs[4].toInt();
    if (thresh == 0)
       thresh = 10000;
    vector<DMatch> matches;
        int dim = desc1.cols/8;
    for (int i = 0; i < desc1.rows; ++i)
    {
        int bestIdx = -1;
        int largestDist = 10000;
        for (int j=0; j < desc2.rows; ++j)
        {
            int dist = matchHammingMasked(
                    desc1.ptr<unsigned __int64>(i),
                    desc2.ptr<unsigned __int64>(j),
                    mask1.ptr<unsigned __int64>(i),
                    mask2.ptr<unsigned __int64>(j),
                    mask1.row(i), mask2.row(j), 
                    dim);
            if (dist < largestDist && dist < thresh)
            {
                largestDist = dist;
                bestIdx = j;
            }
        }
        matches.push_back(DMatch(i, bestIdx, 0, (float)largestDist));  
    }
    plhs[0] = MxArray(matches);
}