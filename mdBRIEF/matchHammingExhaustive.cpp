/**
 * @file matchHamming_binboost.cpp
 * @brief mex interface for BinBoost
 * @author Steffen Urban
 * @date 2015
 */
#include "mexopencv.hpp"

using namespace std;
using namespace cv;

int matchHamming(const Mat& descriptors1,
                 const Mat& descriptors2)
{

	const unsigned __int64* descr_i = descriptors1.ptr<unsigned __int64>(0);
	const unsigned __int64* descr_j = descriptors2.ptr<unsigned __int64>(0);

    unsigned __int64 dist = 0;
    for (int i = 0; i < descriptors1.cols/8; i++) 
    {
        unsigned __int64 axorb = descr_i[i] ^ descr_j[i];
        dist += __popcnt64(axorb);
    }
    
    return (int)dist;
}


void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
    // Argument vector
    vector<MxArray> rhs(prhs,prhs+nrhs);

    Mat desc1 = rhs[0].toMat(CV_8UC1);
    Mat desc2 = rhs[1].toMat(CV_8UC1);
    int thresh = rhs[2].toInt();
    if (thresh == 0)
           thresh = 10000;
    vector<DMatch> matches;
    for (int i = 0; i < desc1.rows; ++i)
    {
        int bestIdx = -1;
        int largestDist = 10000;
        for (int j = 0; j < desc2.rows; ++j)
        {
            int dist = (int)cv::norm(desc1.row(i),desc2.row(j), NORM_HAMMING,Mat());
//             int dist = matchHamming(desc1.row(i),desc2.row(j));
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