#include "mexopencv.hpp"
#include "bold.h"
#include <fstream>
#include <iostream>;
using namespace std;
using namespace cv;



/* load tests and init 2 rotations
   for fast affine aprox. (example -20,20) */
BOLD::BOLD(float angle_patch)
{
    bin_tests = (int**) malloc(DIMS * sizeof(int *));
    for (int i = 0; i < NROTS; i++)
        bin_tests[i] = (int*)malloc(DIMS*2 * sizeof(int));
    
  rotations[0] = 20; 
  rotations[1] = -20;
  int j=0; 
  /* compute the rotations offline */
  for (int i = 0; j < 2*DIMS; i+=4, j+=2) {
    int x1 = learned_bold_pattern_64_[i];
    int y1 = learned_bold_pattern_64_[i+1];
    int x2 = learned_bold_pattern_64_[i+2];
    int y2 = learned_bold_pattern_64_[i+3];

    float ca1 = (float)cos(angle_patch);
    float sa1 = (float)sin(angle_patch);
    
    int x1_r = x1*ca1 - y1*sa1;
    int y1_r = x1*sa1 + y1*ca1;
    int x2_r = x2*ca1 - y2*sa1;
    int y2_r = x2*sa1 + y2*ca1;
    
    // save rotated test
    bin_tests[0][j] = (x1_r + 16)+32*(y1_r + 16); 
    bin_tests[0][j+1] = (x2_r + 16)+32*(y2_r + 16);
    
    for (int a = 1; a < NROTS; a++) {
      float angdeg = rotations[a-1];
      float angle = angdeg*(float)(CV_PI/180.f);
      float ca = (float)cos(angle);
      float sa = (float)sin(angle);
      int rotx1 = (x1_r)*ca - (y1_r)*sa + 16;
      int roty1 = (x1_r)*sa + (y1_r)*ca + 16;
      int rotx2 = (x2_r)*ca - (y2_r)*sa + 16;
      int roty2 = (x2_r)*sa + (y2_r)*ca + 16;
      bin_tests[a][j] = rotx1 + 32*roty1;
      bin_tests[a][j+1] = rotx2 + 32*roty2;
    }    
  }
}

BOLD::~BOLD(void)
{
  /* free the tests */
  for (int i = 0; i < NROTS; i++){
    free(bin_tests[i]);
  }
  free(bin_tests);
}

void BOLD::compute_patch(cv::Mat img, cv::Mat& descr,cv::Mat& masks) 
{
  /* init cv mats */
  int nkeypoints = 1;
  descr.create(nkeypoints, DIMS/8, CV_8U);
  masks.create(nkeypoints, DIMS/8, CV_8U);

  /* apply box filter */  
//   cv::Mat patch;
//   boxFilter(img, patch, img.depth(), cv::Size(5,5),
// 	    cv::Point(-1,-1), true, cv::BORDER_REFLECT);

  /* get test and mask results  */    
  int k =0;
  uchar* dsc = descr.ptr<uchar>(k);
  uchar* msk = masks.ptr<uchar>(k);
  uchar *smoothed = img.data;
  int* tests = bin_tests[0];
  int* r0 = bin_tests[1];
  int* r1 = bin_tests[2];
  unsigned int val = 0;
  unsigned int var = 0;
  int idx = 0;
  int j=0;
  int bit;
  int tdes,tvar;
  for (int i = 0; i < DIMS; i++,j+=2)
  {
    bit = i%8;
    int temp_var=0;
    tdes = (smoothed[tests[j]] < smoothed[tests[j+1]]);    
    temp_var += (smoothed[r0[j]] < smoothed[r0[j+1]])^tdes;
    temp_var += (smoothed[r1[j]] < smoothed[r1[j+1]])^tdes;
    /* tvar-> 0 not stable --------  tvar-> 1 stable */
    tvar = (temp_var == 0) ;
    if (bit==0)
      {
	val = tdes;
	var = tvar;
      } 
    else
      {
	val |= tdes << bit;
	var |= tvar << bit;
      }
    if (bit==7)
      {
	dsc[idx] = val;
	msk[idx] = var;
	val = 0;
	var = 0;
	idx++;
      }
  }
}    
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
    // Argument vector
    vector<MxArray> rhs(prhs,prhs+nrhs);

    Mat patch = rhs[0].toMat(CV_8UC1);
    float angle = rhs[1].toFloat();
    BOLD bold(angle);
    cv::Mat desc_bold,mask_bold;
    bold.compute_patch(patch, desc_bold, mask_bold) ;

    plhs[0] = MxArray(desc_bold);
    plhs[1] = MxArray(mask_bold);
}
