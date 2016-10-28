
#include <fstream>
#include <iostream>

#include <algorithm>
#include "mexopencv.hpp"
#include "mdBRIEF.h"
using namespace cv;
using namespace std;

	/////////// functions
	/**
	* Function that computes the Harris responses in a
	* blockSize x blockSize patch at given points in the image
	*/
	static void HarrisResponses(const Mat& img, 
		const std::vector<Rect>& layerinfo,
		std::vector<KeyPoint>& pts, 
		int blockSize, 
		float harris_k)
	//static void	HarrisResponses(const std::vector<Mat>& img, const std::vector<Rect>& layerinfo,
	//	std::vector<KeyPoint>& pts, int blockSize, float harris_k)
	{
		CV_Assert(img.type() == CV_8UC1 && blockSize*blockSize <= 2048);

		size_t ptidx, ptsize = pts.size();

		const uchar* ptr00 = img.ptr<uchar>();
		int step = (int)(img.step / img.elemSize1());
		int r = blockSize / 2;

		float scale = 1.f / ((1 << 2) * blockSize * 255.f);
		float scale_sq_sq = scale * scale * scale * scale;

		AutoBuffer<int> ofsbuf(blockSize*blockSize);
		int* ofs = ofsbuf;
		for (int i = 0; i < blockSize; i++)
			for (int j = 0; j < blockSize; j++)
				ofs[i*blockSize + j] = (int)(i*step + j);

		for (ptidx = 0; ptidx < ptsize; ptidx++)
		{
			int x0 = cvRound(pts[ptidx].pt.x);
			int y0 = cvRound(pts[ptidx].pt.y);
			int z = pts[ptidx].octave;

			const uchar* ptr0 = ptr00 + (y0 - r + layerinfo[z].y)*step + x0 - r + layerinfo[z].x;
			int a = 0, b = 0, c = 0;

			for (int k = 0; k < blockSize*blockSize; k++)
			{
				const uchar* ptr = ptr0 + ofs[k];
				int Ix = (ptr[1] - ptr[-1]) * 2 + (ptr[-step + 1] - ptr[-step - 1]) + (ptr[step + 1] - ptr[step - 1]);
				int Iy = (ptr[step] - ptr[-step]) * 2 + (ptr[step - 1] - ptr[-step - 1]) + (ptr[step + 1] - ptr[-step + 1]);
				a += Ix*Ix;
				b += Iy*Iy;
				c += Ix*Iy;
			}
			pts[ptidx].response = ((float)a * b - (float)c * c -
				harris_k * ((float)a + b) * ((float)a + b))*scale_sq_sq;
		}
	}
	static void ICAngles(const Mat& img, const std::vector<Rect>& layerinfo,
		std::vector<KeyPoint>& pts, const std::vector<int> & u_max, int half_k)
	{
		int step = (int)img.step1();
		size_t ptidx, ptsize = pts.size();

		for (ptidx = 0; ptidx < ptsize; ptidx++)
		{
			const Rect& layer = layerinfo[pts[ptidx].octave];
			const uchar* center = &img.at<uchar>(cvRound(pts[ptidx].pt.y) + layer.y, cvRound(pts[ptidx].pt.x) + layer.x);

			int m_01 = 0, m_10 = 0;

			// Treat the center line differently, v=0
			for (int u = -half_k; u <= half_k; ++u)
				m_10 += u * center[u];

			// Go line by line in the circular patch
			for (int v = 1; v <= half_k; ++v)
			{
				// Proceed over the two lines
				int v_sum = 0;
				int d = u_max[v];
				for (int u = -d; u <= d; ++u)
				{
					int val_plus = center[u + v*step], val_minus = center[u - v*step];
					v_sum += (val_plus - val_minus);
					m_10 += u * (val_plus + val_minus);
				}
				m_01 += v * v_sum;
			}

			pts[ptidx].angle = fastAtan2((float)m_01, (float)m_10);
		}
	}
	//static void ICAngles(const Mat& img,
	//	const std::vector<Rect>& layerinfo,
	//	std::vector<KeyPoint>& pts,
	//	const std::vector<int> & u_max,
	//	int half_k)
	//static void ICAngles(const std::vector<Mat>& img,
	//	const std::vector<Rect>& layerinfo,
	//	std::vector<KeyPoint>& pts,
	//	const std::vector<int> & u_max,
	//	int half_k)
	//{
	//	
	//	size_t ptidx, ptsize = pts.size();

	//	for (ptidx = 0; ptidx < ptsize; ptidx++)
	//	{
	//		const Rect& layer = layerinfo[pts[ptidx].octave];
	//		int step = (int)img[pts[ptidx].octave].step1();
	//		//const uchar* center =
	//		//	&img.at<uchar>(cvRound(pts[ptidx].pt.y) + layer.y,
	//		//	cvRound(pts[ptidx].pt.x) + layer.x);

	//		const uchar* center = img[pts[ptidx].octave].ptr<uchar>(cvRound(pts[ptidx].pt.y));

	//		int m_01 = 0, m_10 = 0;

	//		// Treat the center line differently, v=0
	//		for (int u = -half_k; u <= half_k; ++u)
	//		{
	//			m_10 += u * center[u];
	//		}

	//		// Go line by line in the circular patch
	//		for (int v = 1; v <= half_k; ++v)
	//		{
	//			// Proceed over the two lines
	//			int v_sum = 0;
	//			int d = u_max[v];
	//			for (int u = -d; u <= d; ++u)
	//			{
	//				const uchar* center1 = img[pts[ptidx].octave].ptr<uchar>(cvRound(pts[ptidx].pt.y)+v);
	//				const uchar* center2 = img[pts[ptidx].octave].ptr<uchar>(cvRound(pts[ptidx].pt.y)- v);
	//				int val_plus = center1[u], val_minus = center2[u];
	//				v_sum += (val_plus - val_minus);
	//				m_10 += u * (val_plus + val_minus);
	//			}
	//			m_01 += v * v_sum;
	//		}

	//		pts[ptidx].angle = fastAtan2((float)m_01, (float)m_10);
	//	}
	//}

	void rotateAndDistortPattern(const Point2f& undist_kps,
		const std::vector<Point>& patternIn, 
		std::vector<Point>& patternOut,
		cCamModelGeneral& camModel,
		const float& ax, const float& ay)
	{
		const int npoints = (int)patternIn.size();
		std::vector<float> xcoords(npoints);
		std::vector<float> ycoords(npoints);
		double sumX = 0.0;
		double sumY = 0.0;
        //float scale = sqrt(pow(undist_kps.x,2)+pow(undist_kps.y,2)) / sqrt(754*754+480*480);
        //if (scale <= 1.0f)
        //    scale = 1.0f;
		for (int p = 0; p < npoints; ++p)
		{
			// rotate pattern point and move it to the keypoint
			float xr = (patternIn[p].x)*ax - (patternIn[p].y)*ay + undist_kps.x;
			float yr = (patternIn[p].x)*ay + (patternIn[p].y)*ax + undist_kps.y;

			Vec2f distorted(0,0);
			camModel.distortPointsOcam(Vec2f(xr, yr), distorted);

			xcoords[p] = distorted(0);
			ycoords[p] = distorted(1);
			sumX += distorted(0);
			sumY += distorted(1);
		}

		double meanX = sumX / (double)npoints;
		double meanY = sumY / (double)npoints;
		// substract mean, to get correct pattern size
		for (int p = 0; p < npoints; ++p)
		{
			patternOut[p].x = cvRound(xcoords[p] - static_cast<float>(meanX));
			patternOut[p].y = cvRound(ycoords[p] - static_cast<float>(meanY));
		}
	}

	static void compute_dBRIEF_descriptor(const Mat& imagePyramid,
		const std::vector<Rect>& layerInfo,
		const std::vector<float>& layerScale,
		const std::vector<KeyPoint>& keypoints,
		const std::vector<Vec2f>& undistortedKeypoints,
		Mat& descriptors,
		Mat& descriptorMasks,
		const std::vector<Point>& _pattern,
		const int dsize,
		cCamModelGeneral& camModel,
		bool learnmask,
        bool doRotation,
        bool distort = true)
	{
		int step = (int)imagePyramid.step;
		int nkeypoints = (int)keypoints.size();	

		const int npoints = 2 * 8 * dsize; //2*512

		float rot1 = 10;
		float rot2 = -10;
		float rho = static_cast<float>(CV_PI) / 180.0f;
#pragma omp parallel for num_threads(4)
		for (int j = 0; j < nkeypoints; ++j)
		{
			std::vector<Point> distortedRotatedPattern(npoints);
			std::vector<std::vector<Point>> maskPattern(2, std::vector<Point>(npoints));
            float angle = 0.0f;
            if (doRotation)
                angle = keypoints[j].angle;
            if (distort)
                rotateAndDistortPattern(undistortedKeypoints[j], _pattern,
                    distortedRotatedPattern, camModel, cos(rho*angle),sin(rho*angle));
            else
            {  
                float ax = cos(rho*angle);
                float ay = sin(rho*angle);
                for (int p = 0; p < npoints; ++p)
                {
                    distortedRotatedPattern[p].x = (_pattern[p].x)*ax - (_pattern[p].y)*ay;
                    distortedRotatedPattern[p].y = (_pattern[p].x)*ay + (_pattern[p].y)*ax;
                }
            }
             
            if (learnmask)
			{
                float aRot1 = rho*(angle + rot1);
                float aRot2 = rho*(angle + rot2);
                Vec2f undistKp1 = undistortedKeypoints[j];
                Vec2f undistKp2 = undistortedKeypoints[j];
				rotateAndDistortPattern(undistKp1, _pattern,
					maskPattern[0], camModel, 
					cos(aRot1), sin(aRot1));
				rotateAndDistortPattern(undistKp2, _pattern,
					maskPattern[1], camModel, 
					cos(aRot2), sin(aRot2));
			}

			const KeyPoint& kpt = keypoints[j];
			const Rect& layer = layerInfo[kpt.octave];
			float scale = 1.f / layerScale[kpt.octave];


			int layerpscalex = cvRound(kpt.pt.x*scale + layer.x);
			int layerpscaley = cvRound(kpt.pt.y*scale + layer.y);
			const uchar* center = 0;

			int ix = 0, iy = 0;
			// get the rotated and distorted pattern for keypoint j
            const Point* pattern = NULL;
  
            pattern = &distortedRotatedPattern[0];
			const Point* maskPattern1 = &maskPattern[0][0];
			const Point* maskPattern2 = &maskPattern[1][0];

			uchar* desc = descriptors.ptr<uchar>(j);
			uchar* descMask = descriptorMasks.ptr<uchar>(j);

#define GET_VALUE(idx) \
               (ix = pattern[idx].x, \
                iy = pattern[idx].y, \
				center = imagePyramid.ptr<uchar>(layerpscaley+iy),\
                center[layerpscalex+ix] )
#define GET_VALUE_MASK1(idx) \
				(ix = maskPattern1[idx].x, \
                iy = maskPattern1[idx].y, \
				center = imagePyramid.ptr<uchar>(layerpscaley+iy),\
                center[layerpscalex+ix] )
#define GET_VALUE_MASK2(idx) \
				(ix = maskPattern2[idx].x, \
                iy = maskPattern2[idx].y, \
				center = imagePyramid.ptr<uchar>(layerpscaley+iy),\
                center[layerpscalex+ix] )

			if (!learnmask)
			{
				for (int i = 0; i < dsize; ++i, pattern += 16)
				{
					int t0, t1, val;
					t0 = GET_VALUE(0); t1 = GET_VALUE(1);
					val = t0 < t1;
					t0 = GET_VALUE(2); t1 = GET_VALUE(3);
					val |= (t0 < t1) << 1;
					t0 = GET_VALUE(4); t1 = GET_VALUE(5);
					val |= (t0 < t1) << 2;
					t0 = GET_VALUE(6); t1 = GET_VALUE(7);
					val |= (t0 < t1) << 3;
					t0 = GET_VALUE(8); t1 = GET_VALUE(9);
					val |= (t0 < t1) << 4;
					t0 = GET_VALUE(10); t1 = GET_VALUE(11);
					val |= (t0 < t1) << 5;
					t0 = GET_VALUE(12); t1 = GET_VALUE(13);
					val |= (t0 < t1) << 6;
					t0 = GET_VALUE(14); t1 = GET_VALUE(15);
					val |= (t0 < t1) << 7;

					desc[i] = (uchar)val;
				}
			}
			else
			{
				for (int i = 0; i < dsize; ++i, pattern += 16,
						maskPattern1 += 16, maskPattern2 += 16)
				{
					int temp_val;
					int t0 , t1 , val, maskVal;
					int mask1_1, mask1_2, mask2_1, mask2_2 , stable_val = 0;
					// first bit
					t0 = GET_VALUE(0); t1 = GET_VALUE(1);
					temp_val = t0 < t1;
					val = temp_val;
					mask1_1 = GET_VALUE_MASK1(0); mask1_2 = GET_VALUE_MASK1(1); // mask1
					mask2_1 = GET_VALUE_MASK2(0); mask2_2 = GET_VALUE_MASK2(1); // mask2
					stable_val += (mask1_1 < mask1_2) ^ temp_val;
					stable_val += (mask2_1 < mask2_2) ^ temp_val;
					maskVal = (stable_val == 0); 
					stable_val = 0;
					// second bit
					t0 = GET_VALUE(2); t1 = GET_VALUE(3);
					temp_val = t0 < t1;
					val |= temp_val << 1;
					mask1_1 = GET_VALUE_MASK1(2); mask1_2 = GET_VALUE_MASK1(3); // mask1
					mask2_1 = GET_VALUE_MASK2(2); mask2_2 = GET_VALUE_MASK2(3); // mask2
					stable_val += (mask1_1 < mask1_2) ^ temp_val;
					stable_val += (mask2_1 < mask2_2) ^ temp_val;
					maskVal |= (stable_val == 0) << 1; 
					stable_val = 0;
					// third bit
					t0 = GET_VALUE(4); t1 = GET_VALUE(5);
					temp_val = t0 < t1;
					val |= temp_val << 2;
					mask1_1 = GET_VALUE_MASK1(4); mask1_2 = GET_VALUE_MASK1(5); // mask1
					mask2_1 = GET_VALUE_MASK2(4); mask2_2 = GET_VALUE_MASK2(5); // mask2
					stable_val += (mask1_1 < mask1_2) ^ temp_val;
					stable_val += (mask2_1 < mask2_2) ^ temp_val;
					maskVal |= (stable_val == 0) << 2; 
					stable_val = 0;
					// fourth bit
					t0 = GET_VALUE(6); t1 = GET_VALUE(7);
					temp_val = t0 < t1;
					val |= temp_val << 3;
					mask1_1 = GET_VALUE_MASK1(6); mask1_2 = GET_VALUE_MASK1(7); // mask1
					mask2_1 = GET_VALUE_MASK2(6); mask2_2 = GET_VALUE_MASK2(7); // mask2
					stable_val += (mask1_1 < mask1_2) ^ temp_val;
					stable_val += (mask2_1 < mask2_2) ^ temp_val;
					maskVal |= (stable_val == 0) << 3; 
					stable_val = 0;
					// fifth bit
					t0 = GET_VALUE(8); t1 = GET_VALUE(9);
					temp_val = t0 < t1;
					val |= temp_val << 4;
					mask1_1 = GET_VALUE_MASK1(8); mask1_2 = GET_VALUE_MASK1(9); // mask1
					mask2_1 = GET_VALUE_MASK2(8); mask2_2 = GET_VALUE_MASK2(9); // mask2
					stable_val += (mask1_1 < mask1_2) ^ temp_val;
					stable_val += (mask2_1 < mask2_2) ^ temp_val;
					maskVal |= (stable_val == 0) << 4;
					stable_val = 0;
					// sixth bit
					t0 = GET_VALUE(10); t1 = GET_VALUE(11);
					temp_val = t0 < t1;
					val |= temp_val << 5;
					mask1_1 = GET_VALUE_MASK1(10); mask1_2 = GET_VALUE_MASK1(11); // mask1
					mask2_1 = GET_VALUE_MASK2(10); mask2_2 = GET_VALUE_MASK2(11); // mask2
					stable_val += (mask1_1 < mask1_2) ^ temp_val;
					stable_val += (mask2_1 < mask2_2) ^ temp_val;
					maskVal |= (stable_val == 0) << 5;
					stable_val = 0;
					// seventh bit
					t0 = GET_VALUE(12); t1 = GET_VALUE(13);
					temp_val = t0 < t1;
					val |= temp_val << 6;
					mask1_1 = GET_VALUE_MASK1(12); mask1_2 = GET_VALUE_MASK1(13); // mask1
					mask2_1 = GET_VALUE_MASK2(12); mask2_2 = GET_VALUE_MASK2(13); // mask2
					stable_val += (mask1_1 < mask1_2) ^ temp_val;
					stable_val += (mask2_1 < mask2_2) ^ temp_val;
					maskVal |= (stable_val == 0) << 6;
					stable_val = 0;
					// eigth bit
					t0 = GET_VALUE(14); t1 = GET_VALUE(15);
					temp_val = t0 < t1;
					val |= temp_val << 7;
					mask1_1 = GET_VALUE_MASK1(14); mask1_2 = GET_VALUE_MASK1(15); // mask1
					mask2_1 = GET_VALUE_MASK2(14); mask2_2 = GET_VALUE_MASK2(15); // mask2
					stable_val += (mask1_1 < mask1_2) ^ temp_val;
					stable_val += (mask2_1 < mask2_2) ^ temp_val;
					maskVal |= (stable_val == 0) << 7;

					desc[i] = (uchar)val;
					descMask[i] = (uchar)maskVal;
				}
			}

#undef GET_VALUE
#undef GET_VALUE_MASK1
#undef GET_VALUE_MASK2
		}
	}

	static void initialize_mdBRIEF_Pattern(const Point* pattern0,
		std::vector<Point>& pattern,
		int ntuples,
		int tupleSize,
		int poolSize)
	{
		RNG rng(0x12345678);
		int i, k, k1;
		pattern.resize(ntuples*tupleSize);

		for (i = 0; i < ntuples; i++)
		{
			for (k = 0; k < tupleSize; k++)
			{
				for (;;)
				{
					int idx = rng.uniform(0, poolSize);
					Point pt = pattern0[idx];
					for (k1 = 0; k1 < k; k1++)
						if (pattern[tupleSize*i + k1] == pt)
							break;
					if (k1 == k)
					{
						pattern[tupleSize*i + k] = pt;
						break;
					}
				}
			}
		}
	}

	static void makeRandomPattern(int patchSize,
		Point* pattern,
		int npoints)
	{
		RNG rng(0x34985739); // we always start with a fixed seed,
		// to make patterns the same on each run
		for (int i = 0; i < npoints; ++i)
		{
			pattern[i].x = rng.uniform(-patchSize / 2, patchSize / 2 + 1);
			pattern[i].y = rng.uniform(-patchSize / 2, patchSize / 2 + 1);
		}
	}

	static inline float getScale(int level,int firstLevel,double scaleFactor)
	{
		return (float)std::pow(scaleFactor, (double)(level - firstLevel));
	}

	/** Compute the ORB_Impl keypoints on an image
	* @param image_pyramid the image pyramid to compute the features and descriptors on
	* @param mask_pyramid the masks to apply at every level
	* @param keypoints the resulting keypoints, clustered per level
	*/
	static void computeKeyPoints(const Mat& imagePyramid,
		const Mat& maskPyramid,
		const std::vector<Rect>& layerInfo,
		const std::vector<float>& layerScale,
		cCamModelGeneral& camModel,
		std::vector<KeyPoint>& allKeypoints,
		std::vector<Vec2f>& undistortedKeypoints,
		int nfeatures,
		double scaleFactor,
		int edgeThreshold,
		int patchSize,
		int scoreType,
		int fastThreshold,
		bool useAgast,
		int fastAgastType)
	{

		int i, nkeypoints, level, nlevels = (int)layerInfo.size();
		std::vector<int> nfeaturesPerLevel(nlevels);

		// fill the extractors and descriptors for the corresponding scales
		float factor = (float)(1.0 / scaleFactor);
		float ndesiredFeaturesPerScale = nfeatures*(1 - factor) /
			(1 - (float)std::pow((double)factor, (double)nlevels));

		int sumFeatures = 0;
		for (level = 0; level < nlevels - 1; level++)
		{
			nfeaturesPerLevel[level] = cvRound(ndesiredFeaturesPerScale);
			sumFeatures += nfeaturesPerLevel[level];
			ndesiredFeaturesPerScale *= factor;
		}
		nfeaturesPerLevel[nlevels - 1] = max(nfeatures - sumFeatures, 0);

		// Make sure we forget about what is too close to the boundary
		//edge_threshold_ = max(edge_threshold_, patch_size_/2 + kKernelWidth / 2 + 2);

		// pre-compute the end of a row in a circular patch
		int halfPatchSize = patchSize / 2;
		std::vector<int> umax(halfPatchSize + 2);

		int v, v0, vmax = cvFloor(halfPatchSize * std::sqrt(2.f) / 2 + 1);
		int vmin = cvCeil(halfPatchSize * std::sqrt(2.f) / 2);
		for (v = 0; v <= vmax; ++v)
			umax[v] = cvRound(std::sqrt((double)halfPatchSize * halfPatchSize - v * v));

		// Make sure we are symmetric
		for (v = halfPatchSize, v0 = 0; v >= vmin; --v)
		{
			while (umax[v0] == umax[v0 + 1])
				++v0;
			umax[v] = v0;
			++v0;
		}

		allKeypoints.clear();
		std::vector<KeyPoint> keypoints;
		std::vector<int> counters(nlevels);
		keypoints.reserve(nfeaturesPerLevel[0] * 2);

		for (level = 0; level < nlevels; level++)
		{
			int featuresNum = nfeaturesPerLevel[level];
			Mat img = imagePyramid(layerInfo[level]);
			Mat mask = maskPyramid.empty() ? Mat() : maskPyramid(layerInfo[level]);

			// Detect FAST features, 20 is a good threshold
			{
				if (useAgast)
				{
					Ptr<AgastFeatureDetector> ag = AgastFeatureDetector::create(fastThreshold, true, fastAgastType);
					ag->detect(img, keypoints, mask);
				}
				else
				{
					Ptr<FastFeatureDetector> fd = FastFeatureDetector::create(fastThreshold, true, fastAgastType);
					fd->detect(img, keypoints, mask);
				}
			}

			// Remove keypoints very close to the border
			KeyPointsFilter::runByImageBorder(keypoints, img.size(), edgeThreshold);

			// Keep more points than necessary as FAST does not give amazing corners
			KeyPointsFilter::retainBest(keypoints, scoreType == HARRIS_SCORE_ ? 2 * featuresNum : featuresNum);

			nkeypoints = (int)keypoints.size();
			counters[level] = nkeypoints;

			float sf = layerScale[level];
			for (i = 0; i < nkeypoints; i++)
			{
				keypoints[i].octave = level;
				keypoints[i].size = patchSize*sf;
			}

			std::copy(keypoints.begin(), keypoints.end(), std::back_inserter(allKeypoints));
		}

		nkeypoints = (int)allKeypoints.size();
		if (nkeypoints == 0)
		{
			return;
		}
		Mat responses;

		// Select best features using the Harris cornerness (better scoring than FAST)
		//if (scoreType == HARRIS_SCORE_)
		//{
			HarrisResponses(imagePyramid, layerInfo, allKeypoints, 7, HARRIS_K_);

			std::vector<KeyPoint> newAllKeypoints;
			newAllKeypoints.reserve(nfeaturesPerLevel[0] * nlevels);

			int offset = 0;
			for (level = 0; level < nlevels; level++)
			{
				int featuresNum = nfeaturesPerLevel[level];
				nkeypoints = counters[level];
				keypoints.resize(nkeypoints);
				std::copy(allKeypoints.begin() + offset,
					allKeypoints.begin() + offset + nkeypoints,
					keypoints.begin());
				offset += nkeypoints;

				//cull to the final desired level, using the new Harris scores.
				KeyPointsFilter::retainBest(keypoints, featuresNum);

				std::copy(keypoints.begin(), keypoints.end(), std::back_inserter(newAllKeypoints));
			}
			std::swap(allKeypoints, newAllKeypoints);
		//}

		nkeypoints = (int)allKeypoints.size();

		ICAngles(imagePyramid, layerInfo, allKeypoints, umax, halfPatchSize);

		for (i = 0; i < nkeypoints; i++)
		{
			float scale = layerScale[allKeypoints[i].octave];
			allKeypoints[i].pt *= scale;
		}

		//////////////////
		// undistort keypoints
		//////////////////
		undistortedKeypoints = std::vector<Vec2f>(nkeypoints);
		const double scaleF = camModel.Get_P().at<double>(0);
		for (int i = 0; i < nkeypoints; ++i)
			camModel.undistortPointsOcam(allKeypoints[i].pt, scaleF, undistortedKeypoints[i]);
	}

	int mdBRIEF_Impl::descriptorSize() const
	{
		return descSize;
	}

	int mdBRIEF_Impl::descriptorType() const
	{
		return CV_8U;
	}

	int mdBRIEF_Impl::defaultNorm() const
	{
		return NORM_HAMMING;
	}

	void mdBRIEF_Impl::detectAndCompute(InputArray _image, 
		InputArray _mask,
		std::vector<KeyPoint>& keypoints,
		OutputArray _descriptors,
		OutputArray _descriptorMasks,
		cCamModelGeneral& camModel,
		bool useProvidedKeypoints)
	{
		CV_Assert(patchSize >= 2);

		bool do_keypoints = !useProvidedKeypoints;
		bool do_descriptors = _descriptors.needed();
		bool do_masks = learnMasks;

		if ((!do_keypoints && !do_descriptors) || _image.empty())
			return;

		//ROI handling
		const int HARRIS_BLOCK_SIZE = 9;
		int halfPatchSize = patchSize / 2;
		int border = max(edgeThreshold, max(halfPatchSize, HARRIS_BLOCK_SIZE / 2)) + 1;

		Mat image = _image.getMat(), mask = _mask.getMat();
		if (image.type() != CV_8UC1)
			cvtColor(_image, image, COLOR_BGR2GRAY);

		int i, level, nLevels = this->numlevels, nkeypoints = (int)keypoints.size();
		bool sortedByLevel = true;

		if (!do_keypoints)
		{
			// if we have pre-computed keypoints, they may use more levels than it is set in parameters
			// !!!TODO!!! implement more correct method, independent from the used keypoint detector.
			// Namely, the detector should provide correct size of each keypoint. Based on the keypoint size
			// and the algorithm used (i.e. BRIEF, running on 31x31 patches) we should compute the approximate
			// scale-factor that we need to apply. Then we should cluster all the computed scale-factors and
			// for each cluster compute the corresponding image.
			//
			// In short, ultimately the descriptor should
			// ignore octave parameter and deal only with the keypoint size.
			nLevels = 0;
			for (i = 0; i < nkeypoints; i++)
			{
				level = keypoints[i].octave;
				CV_Assert(level >= 0);
				if (i > 0 && level < keypoints[i - 1].octave)
					sortedByLevel = false;
				nLevels = max(nLevels, level);
			}
			nLevels++;
		}


		std::vector<cv::Rect> layerInfo(nLevels);
		std::vector<int> layerOfs(nLevels);
		std::vector<float> layerScale(nLevels);

		int level_dy = image.rows + border * 2;
		Point level_ofs(0, 0);
		Size bufSize((image.cols + border * 2 + 15) & -16, 0);

		for (level = 0; level < nLevels; level++)
		{
			float scale = getScale(level, firstLevel, scaleFactor);
			layerScale[level] = scale;
			Size sz(cvRound(image.cols / scale), cvRound(image.rows / scale));
			Size wholeSize(sz.width + border * 2, sz.height + border * 2);
			if (level_ofs.x + wholeSize.width > bufSize.width)
			{
				level_ofs = Point(0, level_ofs.y + level_dy);
				level_dy = wholeSize.height;
			}

			Rect linfo(level_ofs.x + border, level_ofs.y + border, sz.width, sz.height);
			layerInfo[level] = linfo;
			layerOfs[level] = linfo.y*bufSize.width + linfo.x;
			level_ofs.x += wholeSize.width;
		}
		bufSize.height = level_ofs.y + level_dy;

		Mat imagePyramid, maskPyramid;
		imagePyramid.create(bufSize, CV_8U);
		if (!mask.empty())
			maskPyramid.create(bufSize, CV_8U);

		Mat prevImg = image, prevMask = mask;

		//this->imgPyr = std::vector<Mat>(nLevels);
		//this->imgPyrSmoothed = std::vector<Mat>(nLevels);
		//for (level = 0; level < nLevels; ++level)
		//{
		//	Mat currImg;
		//	float s = 1 / getScale(level, firstLevel, scaleFactor);
		//	cv::resize(image, currImg, Size(0, 0), s, s, INTER_LINEAR);
		//	imgPyr[level] = currImg;

		//	cv::GaussianBlur(currImg, this->imgPyrSmoothed[level], Size(7, 7), 2.0, 2.0, BORDER_DEFAULT);
		//	Mat test = this->imgPyrSmoothed[level];
		//}

		// Pre-compute the scale pyramids
		for (level = 0; level < nLevels; ++level)
		{
			Rect linfo = layerInfo[level];
			Size sz(linfo.width, linfo.height);
			Size wholeSize(sz.width + border * 2, sz.height + border * 2);
			Rect wholeLinfo = Rect(linfo.x - border, linfo.y - border, wholeSize.width, wholeSize.height);
			Mat extImg = imagePyramid(wholeLinfo), extMask;
			Mat currImg = extImg(Rect(border, border, sz.width, sz.height)), currMask;

			if (!mask.empty())
			{
				extMask = maskPyramid(wholeLinfo);
				currMask = extMask(Rect(border, border, sz.width, sz.height));
			}

			// Compute the resized image
			if (level != firstLevel)
			{
				resize(prevImg, currImg, sz, 0, 0, INTER_LINEAR);
				if (!mask.empty())
				{
					resize(prevMask, currMask, sz, 0, 0, INTER_LINEAR);
					if (level > firstLevel)
						threshold(currMask, currMask, 254, 0, THRESH_TOZERO);
				}

				copyMakeBorder(currImg, extImg, border, border, border, border,
					BORDER_REFLECT_101 + BORDER_ISOLATED);
				if (!mask.empty())
					copyMakeBorder(currMask, extMask, border, border, border, border,
					BORDER_CONSTANT + BORDER_ISOLATED);
			}
			else
			{
				copyMakeBorder(image, extImg, border, border, border, border,
					BORDER_REFLECT_101);
				if (!mask.empty())
					copyMakeBorder(mask, extMask, border, border, border, border,
					BORDER_CONSTANT + BORDER_ISOLATED);
			}
			prevImg = currImg;
			prevMask = currMask;
		}

		//////////////////////
		// do keypoints
		//////////////////////
		std::vector<Vec2f> undistortedKeypoints;
		if (do_keypoints)
		{
			// Get keypoints, those will be far enough from the border that no check will be required for the descriptor
			computeKeyPoints(imagePyramid, maskPyramid,
				layerInfo, layerScale,
				camModel,
				keypoints, undistortedKeypoints,
				nfeatures, scaleFactor, edgeThreshold, patchSize, scoreType, fastThreshold,
				useAgast, fastAgastType); // use agast?
		}
		else
		{
			KeyPointsFilter::runByImageBorder(keypoints, image.size(), edgeThreshold);

			if (!sortedByLevel)
			{
				std::vector<std::vector<KeyPoint> > allKeypoints(nLevels);
				nkeypoints = (int)keypoints.size();
				for (i = 0; i < nkeypoints; i++)
				{
					level = keypoints[i].octave;
					CV_Assert(0 <= level);
					allKeypoints[level].push_back(keypoints[i]);
				}
				keypoints.clear();
				for (level = 0; level < nLevels; level++)
					std::copy(allKeypoints[level].begin(),
					allKeypoints[level].end(),
					std::back_inserter(keypoints));
			}
		}
		//////////////////////
		// do descriptors
		//////////////////////
		if (do_descriptors)
		{
			int dsize = descriptorSize();

			nkeypoints = (int)keypoints.size();
			if (nkeypoints == 0)
			{
				_descriptors.release();
				_descriptorMasks.release();
				return;
			}

			_descriptors.create(nkeypoints, dsize, CV_8UC1);
			_descriptorMasks.create(nkeypoints, dsize, CV_8UC1);

			std::vector<Point> pattern;

			const int npoints = 2 * 8 * dsize; //2*512
			//Point patternbuf[npoints];
            const Point* pattern0;
            switch (testSet)
            {
//                 case 0: pattern0 = (const Point*)learned_orb_pattern_64_; break;
//                 case 1: pattern0 = (const Point*)learned_orb_pattern_32_fisheyeSet; break;
//                 case 2: pattern0 = (const Point*)learned_orb_pattern_32_perspective; break;
//                 case 3: pattern0 = (const Point*)brief_pattern_64_; break;
//                 case 4: pattern0 = (const Point*)learned_pattern_8_OCAM; break;
//                 case 5: pattern0 = (const Point*)learned_pattern_8_orb; break;
//                 case 6: pattern0 = (const Point*)learned_bold_pattern_64_; break;
//                 default: pattern0 = (const Point*)learned_orb_pattern_64_; break;
                case 0: pattern0 = (const Point*)learned_orb_pattern_64_perspective; break;
                case 1: pattern0 = (const Point*)learned_orb_pattern_64_fisheye; break;
                case 2: pattern0 = (const Point*)learned_orb_pattern_64_fisheyeOCAM; break;
                case 3: pattern0 = (const Point*)learned_orb_pattern_64_fisheyeOCAM_corr2; break;
                case 4: pattern0 = (const Point*)learned_surf_pattern_64_perspective; break;
                case 5: pattern0 = (const Point*)learned_surf_pattern_64_fisheye; break;
                case 6: pattern0 = (const Point*)learned_akaze_pattern_64_perspective; break;
                case 7: pattern0 = (const Point*)learned_akaze_pattern_64_fisheye; break;
                case 8: pattern0 = (const Point*)brief_pattern_64_; break;

                default: pattern0 = (const Point*)learned_orb_pattern_64_perspective; break;

            }


			std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

			// smooth images in the pyramid
			for (level = 0; level < nLevels; level++)
			{
				// preprocess the resized image
				Mat workingMat = imagePyramid(layerInfo[level]);
				//boxFilter(workingMat, workingMat, workingMat.depth(), Size(5, 5), Point(-1, -1), true, BORDER_REFLECT_101);
				GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);
			}

			Mat descriptors = _descriptors.getMat();
			Mat descriptorMasks = _descriptorMasks.getMat();
			compute_dBRIEF_descriptor(imagePyramid, layerInfo, layerScale,
				keypoints, undistortedKeypoints,
				descriptors, descriptorMasks,
				pattern, dsize, camModel, learnMasks,doRotation, distort);
		}

	}

	void mdBRIEF_Impl::detect(InputArray _image, 
		InputArray _mask,
		cCamModelGeneral& camModel,
		std::vector<KeyPoint>& keypoints,
		std::vector<Vec2f>& undist_keypoints)
	{
		CV_Assert(patchSize >= 2);

		bool do_keypoints = true;
		bool do_masks = learnMasks;

		if ((!do_keypoints) || _image.empty())
			return;

		//ROI handling
		const int HARRIS_BLOCK_SIZE = 9;
		int halfPatchSize = patchSize / 2;
		int border = max(edgeThreshold, max(halfPatchSize, HARRIS_BLOCK_SIZE / 2)) + 1;

		Mat image = _image.getMat(), mask = _mask.getMat();
		if (image.type() != CV_8UC1)
			cvtColor(_image, image, COLOR_BGR2GRAY);

		int i, level, nLevels = this->numlevels, nkeypoints = (int)keypoints.size();
		bool sortedByLevel = true;

		if (!do_keypoints)
		{
			// if we have pre-computed keypoints, they may use more levels than it is set in parameters
			// !!!TODO!!! implement more correct method, independent from the used keypoint detector.
			// Namely, the detector should provide correct size of each keypoint. Based on the keypoint size
			// and the algorithm used (i.e. BRIEF, running on 31x31 patches) we should compute the approximate
			// scale-factor that we need to apply. Then we should cluster all the computed scale-factors and
			// for each cluster compute the corresponding image.
			//
			// In short, ultimately the descriptor should
			// ignore octave parameter and deal only with the keypoint size.
			nLevels = 0;
			for (i = 0; i < nkeypoints; i++)
			{
				level = keypoints[i].octave;
				CV_Assert(level >= 0);
				if (i > 0 && level < keypoints[i - 1].octave)
					sortedByLevel = false;
				nLevels = max(nLevels, level);
			}
			nLevels++;
		}

		std::vector<cv::Rect> layerInfo(nLevels);
		std::vector<int> layerOfs(nLevels);
		std::vector<float> layerScale(nLevels);

		int level_dy = image.rows + border * 2;
		Point level_ofs(0, 0);
		Size bufSize((image.cols + border * 2 + 15) & -16, 0);

		for (level = 0; level < nLevels; level++)
		{
			float scale = getScale(level, firstLevel, scaleFactor);
			layerScale[level] = scale;
			Size sz(cvRound(image.cols / scale), cvRound(image.rows / scale));
			Size wholeSize(sz.width + border * 2, sz.height + border * 2);
			if (level_ofs.x + wholeSize.width > bufSize.width)
			{
				level_ofs = Point(0, level_ofs.y + level_dy);
				level_dy = wholeSize.height;
			}

			Rect linfo(level_ofs.x + border, level_ofs.y + border, sz.width, sz.height);
			layerInfo[level] = linfo;
			layerOfs[level] = linfo.y*bufSize.width + linfo.x;
			level_ofs.x += wholeSize.width;
		}

		bufSize.height = level_ofs.y + level_dy;

		cv::Mat imagePyramid, maskPyramid;
		imagePyramid.create(bufSize, CV_8U);

		if (!mask.empty())
			maskPyramid.create(bufSize, CV_8U);

		Mat tmp = imagePyramid;
		Mat prevImg = image, prevMask = mask;

		// Pre-compute the scale pyramids
		for (level = 0; level < nLevels; ++level)
		{
			Rect linfo = layerInfo[level];
			Size sz(linfo.width, linfo.height);
			Size wholeSize(sz.width + border * 2, sz.height + border * 2);
			Rect wholeLinfo = Rect(linfo.x - border, linfo.y - border, wholeSize.width, wholeSize.height);
			Mat extImg = imagePyramid(wholeLinfo), extMask;
			Mat currImg = extImg(Rect(border, border, sz.width, sz.height)), currMask;

			if (!mask.empty())
			{
				extMask = maskPyramid(wholeLinfo);
				currMask = extMask(Rect(border, border, sz.width, sz.height));
			}

			// Compute the resized image
			if (level != firstLevel)
			{
				resize(prevImg, currImg, sz, 0, 0, INTER_LINEAR);
				if (!mask.empty())
				{
					resize(prevMask, currMask, sz, 0, 0, INTER_LINEAR);
					if (level > firstLevel)
						threshold(currMask, currMask, 254, 0, THRESH_TOZERO);
				}

				copyMakeBorder(currImg, extImg, border, border, border, border,
					BORDER_REFLECT_101 + BORDER_ISOLATED);
				if (!mask.empty())
					copyMakeBorder(currMask, extMask, border, border, border, border,
					BORDER_CONSTANT + BORDER_ISOLATED);
			}
			else
			{
				copyMakeBorder(image, extImg, border, border, border, border,
					BORDER_REFLECT_101);
				if (!mask.empty())
					copyMakeBorder(mask, extMask, border, border, border, border,
					BORDER_CONSTANT + BORDER_ISOLATED);
			}
			prevImg = currImg;
			prevMask = currMask;
		}

		//////////////////////
		// do keypoints
		//////////////////////

		if (do_keypoints)
		{
			// Get keypoints, those will be far enough from the border that no check will be required for the descriptor
			computeKeyPoints(imagePyramid, maskPyramid,
				layerInfo, layerScale,
				camModel,
				keypoints, undist_keypoints,
				nfeatures, scaleFactor, edgeThreshold, patchSize, scoreType, fastThreshold,
				useAgast, fastAgastType); // use agast?
		}
		else
		{
			KeyPointsFilter::runByImageBorder(keypoints, image.size(), edgeThreshold);

			if (!sortedByLevel)
			{
				std::vector<std::vector<KeyPoint> > allKeypoints(nLevels);
				nkeypoints = (int)keypoints.size();
				for (i = 0; i < nkeypoints; i++)
				{
					level = keypoints[i].octave;
					CV_Assert(0 <= level);
					allKeypoints[level].push_back(keypoints[i]);
				}
				keypoints.clear();
				for (level = 0; level < nLevels; level++)
					std::copy(allKeypoints[level].begin(),
					allKeypoints[level].end(),
					std::back_inserter(keypoints));
			}
		}
	}

	void mdBRIEF_Impl::compute(InputArray _image,
		InputArray _mask,
		const std::vector<KeyPoint>& keypoints,
		const std::vector<Vec2f>& undist_keypoints,
		OutputArray _descriptors,
		OutputArray _descriptorMasks,
		cCamModelGeneral& camModel)
	{
		CV_Assert(patchSize >= 2);

		bool do_descriptors = _descriptors.needed();
		bool do_masks = learnMasks;

		//ROI handling
		const int HARRIS_BLOCK_SIZE = 9;
		int halfPatchSize = patchSize / 2;
		int border = max(edgeThreshold, max(halfPatchSize, HARRIS_BLOCK_SIZE / 2)) + 1;

		Mat image = _image.getMat();
		Mat mask = _mask.getMat();
		if (image.type() != CV_8UC1)
			cvtColor(_image, image, COLOR_BGR2GRAY);

		int level, nLevels = this->numlevels, nkeypoints = (int)keypoints.size();
		bool sortedByLevel = true;

		std::vector<cv::Rect> layerInfo(nLevels);
		std::vector<int> layerOfs(nLevels);
		std::vector<float> layerScale(nLevels);

		int level_dy = image.rows + border * 2;
		Point level_ofs(0, 0);
		Size bufSize((image.cols + border * 2 + 15) & -16, 0);

		for (level = 0; level < nLevels; level++)
		{
			float scale = getScale(level, firstLevel, scaleFactor);
			layerScale[level] = scale;
			Size sz(cvRound(image.cols / scale), cvRound(image.rows / scale));
			Size wholeSize(sz.width + border * 2, sz.height + border * 2);
			if (level_ofs.x + wholeSize.width > bufSize.width)
			{
				level_ofs = Point(0, level_ofs.y + level_dy);
				level_dy = wholeSize.height;
			}

			Rect linfo(level_ofs.x + border, level_ofs.y + border, sz.width, sz.height);
			layerInfo[level] = linfo;
			layerOfs[level] = linfo.y*bufSize.width + linfo.x;
			level_ofs.x += wholeSize.width;
		}

		bufSize.height = level_ofs.y + level_dy;

		cv::Mat imagePyramid, maskPyramid;
		imagePyramid.create(bufSize, CV_8U);

		if (!mask.empty())
			maskPyramid.create(bufSize, CV_8U);

		Mat tmp = imagePyramid;
		Mat prevImg = image, prevMask = mask;

		// Pre-compute the scale pyramids
		for (level = 0; level < nLevels; ++level)
		{
			Rect linfo = layerInfo[level];
			Size sz(linfo.width, linfo.height);
			Size wholeSize(sz.width + border * 2, sz.height + border * 2);
			Rect wholeLinfo = Rect(linfo.x - border, linfo.y - border, wholeSize.width, wholeSize.height);
			Mat extImg = imagePyramid(wholeLinfo), extMask;
			Mat currImg = extImg(Rect(border, border, sz.width, sz.height)), currMask;

			if (!mask.empty())
			{
				extMask = maskPyramid(wholeLinfo);
				currMask = extMask(Rect(border, border, sz.width, sz.height));
			}

			// Compute the resized image
			if (level != firstLevel)
			{
				resize(prevImg, currImg, sz, 0, 0, INTER_LINEAR);
				if (!mask.empty())
				{
					resize(prevMask, currMask, sz, 0, 0, INTER_LINEAR);
					if (level > firstLevel)
						threshold(currMask, currMask, 254, 0, THRESH_TOZERO);
				}

				copyMakeBorder(currImg, extImg, border, border, border, border,
					BORDER_REFLECT_101 + BORDER_ISOLATED);
				if (!mask.empty())
					copyMakeBorder(currMask, extMask, border, border, border, border,
					BORDER_CONSTANT + BORDER_ISOLATED);
			}
			else
			{
				copyMakeBorder(image, extImg, border, border, border, border,
					BORDER_REFLECT_101);
				if (!mask.empty())
					copyMakeBorder(mask, extMask, border, border, border, border,
					BORDER_CONSTANT + BORDER_ISOLATED);
			}
			prevImg = currImg;
			prevMask = currMask;
		}

		//////////////////////
		// do descriptors
		//////////////////////
		if (do_descriptors)
		{
			int dsize = descriptorSize();

			nkeypoints = (int)keypoints.size();
			if (nkeypoints == 0)
			{
				_descriptors.release();
				_descriptorMasks.release();
				return;
			}

			_descriptors.create(nkeypoints, dsize, CV_8UC1);
			_descriptorMasks.create(nkeypoints, dsize, CV_8UC1);

			std::vector<Point> pattern;

			const int npoints = 2 * 8 * dsize; 
			//Point patternbuf[npoints];
            const Point* pattern0;
            switch (testSet)
            {
//                 case 0: pattern0 = (const Point*)learned_orb_pattern_64_; break;
//                 case 1: pattern0 = (const Point*)learned_orb_pattern_32_fisheyeSet; break;
//                 case 2: pattern0 = (const Point*)learned_orb_pattern_32_perspective; break;
//                 case 3: pattern0 = (const Point*)brief_pattern_64_; break;
//                 case 4: pattern0 = (const Point*)learned_pattern_8_OCAM; break;
//                 case 5: pattern0 = (const Point*)learned_pattern_8_orb; break;
//                 case 6: pattern0 = (const Point*)learned_bold_pattern_64_; break;
//                 default: pattern0 = (const Point*)learned_orb_pattern_64_; break;
                case 0: pattern0 = (const Point*)learned_orb_pattern_64_perspective; break;
                case 1: pattern0 = (const Point*)learned_orb_pattern_64_fisheye; break;
                case 2: pattern0 = (const Point*)learned_orb_pattern_64_fisheyeOCAM; break;
                case 3: pattern0 = (const Point*)learned_orb_pattern_64_fisheyeOCAM_corr2; break;
                case 4: pattern0 = (const Point*)learned_surf_pattern_64_perspective; break;
                case 5: pattern0 = (const Point*)learned_surf_pattern_64_fisheye; break;
                case 6: pattern0 = (const Point*)learned_akaze_pattern_64_perspective; break;
                case 7: pattern0 = (const Point*)learned_akaze_pattern_64_fisheye; break;
                case 8: pattern0 = (const Point*)brief_pattern_64_; break;

                default: pattern0 = (const Point*)learned_orb_pattern_64_perspective; break;

            }

			std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

			// smooth images in the pyramid
			for (level = 0; level < nLevels; level++)
			{
				//Mat tmp = imagePyramid;
				// preprocess the resized image
				Mat workingMat = imagePyramid(layerInfo[level]);
				//boxFilter(workingMat, workingMat, workingMat.depth(), Size(5, 5), Point(-1, -1), true, BORDER_REFLECT_101);
				GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);
			}

			Mat descriptors = _descriptors.getMat();
			Mat descriptorMasks = _descriptorMasks.getMat();
			compute_dBRIEF_descriptor(imagePyramid, layerInfo, layerScale,
				keypoints, undist_keypoints,
				descriptors, descriptorMasks,
				pattern, dsize, camModel, learnMasks,doRotation,distort);
		}

	}
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
    // Argument vector
    vector<MxArray> rhs(prhs,prhs+nrhs);
    
    cv::Vec<double, 5> cdeu0v0(0.999934215335333,
		-0.026203662446617,
		0.026631962956669,
		3.856408834438496e+02,
		2.407245455764566e+02);
	cv::Mat p(5, 1, CV_64FC1);
	p.at<double>(0) = -209.102840178620;
	p.at<double>(1) = 0.0;
	p.at<double>(2) = 0.00219808982029219;
	p.at<double>(3) = -4.78500789580928e-06;
	p.at<double>(4) = 1.91374184127112e-08;
	cv::Mat invP(10, 1, CV_64FC1);
	invP.at<double>(9) = 3.33467943204825;
	invP.at<double>(8) = 24.5610774120853;
	invP.at<double>(7) = 65.7707853072233;
	invP.at<double>(6) = 74.1358772483720;
	invP.at<double>(5) = 27.1834719956959;
	invP.at<double>(4) = 9.03743234012921;
	invP.at<double>(3) = 26.6018578221928;
	invP.at<double>(2) = -12.2539724533414;
	invP.at<double>(1) = 148.010271268061;
	invP.at<double>(0) = 292.858600545416;
	cCamModelGeneral camModel(cdeu0v0, p, invP);
           
    Mat image(rhs[0].toMat(CV_8U)), descriptors, masks;
    // for orb
	int nFeats = rhs[1].toInt();
	int nLevels = rhs[2].toInt();
    bool learnMask = (bool)rhs[3].toInt();
	int fastType = rhs[4].toInt();
    bool useAgast = (bool)rhs[5].toInt();
	int descSize = (int)rhs[6].toInt();
    descSize = static_cast<int>((float)descSize/8.0f);
    bool doRotation = (bool)rhs[7].toInt();
    int testSet = rhs[8].toInt();
    bool distort = rhs[9].toInt();
    
    float scaleFactor = 1.2;
    int fastThresh = 20;
    int patchSize = 32;
    int edgeThreshold = 31;
    int firstLevel = 0;
    int scoreType = 0;
    
    mdBRIEF_Impl test_mdBrief(nFeats, scaleFactor, nLevels, edgeThreshold,
			firstLevel, scoreType, patchSize, fastThresh, useAgast,
            fastType, learnMask,descSize, doRotation, testSet, distort);
    
    std::vector<KeyPoint> keypoints, undistKeypoints;
    
    test_mdBrief.detectAndCompute(image, Mat(), keypoints, descriptors, masks, camModel, false);


    plhs[0] = MxArray(keypoints);
    if (nlhs > 1)
        plhs[1] = MxArray(descriptors);
    if (nlhs > 2) 
        plhs[2] = MxArray(masks);

}
