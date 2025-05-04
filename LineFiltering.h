#pragma once
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui_c.h>

using namespace cv;
double static const PI_FLOAT = 3.1415926535897932384626433832795;

class LineFiltering
{
public:
	LineFiltering();
	~LineFiltering();
public:
	cv::Mat BRG2Y(cv::Mat img);
	void GetMLD(const cv::Mat& input, cv::Mat& mld, cv::Mat1i& maxDir, int pSize);   //compute  Maximum Local Difference
	void GetMLDwith2x2(const cv::Mat& input, cv::Mat& mld, cv::Mat1i& maxDir);
	cv::Mat MeanFiltering(const cv::Mat& input, int pSize);                // mean  filter
	cv::Mat nonlinearEnhance(const cv::Mat& mld, int pSize, int sigma);           //Eq. 2
	cv::Mat heuristicMP(const cv::Mat input, int t);                             //Eq. 4
	cv::Mat1i InitialScale(const cv::Mat enhanceMP, int k, int minscale, bool isfixminS);
	cv::Mat1i AdjustScale(cv::Mat1i& initialScale, cv::Mat& inputImg, int pSize, const cv::Mat& mld, cv::Mat& final_dir);
	cv::Mat GetGuidance(const cv::Mat input, const cv::Mat1i scale, const cv::Mat1i& final_dir);

	cv::Mat lfHtfilter_o(cv::Mat m_img3f, cv::Mat guide, int psize, float sigma_d, float sigma_r);
	cv::Mat lfVtfilter_o(cv::Mat m_img3f, cv::Mat guide, int psize, float sigma_d, float sigma_r);

	cv::Mat LFHfilter_VariableScale(cv::Mat input, cv::Mat guide, cv::Mat scale, float sigma_d, float sigma_r);
	cv::Mat LFVfilter_VariableScale(cv::Mat input, cv::Mat guide, cv::Mat scale, float sigma_d, float sigma_r);

	cv::Mat bftfilter_o(cv::Mat m_img3f, cv::Mat guide, int ksize, float sigma_d, float sigma_r);
	cv::Mat bftfilter_varyingwindow(cv::Mat input, cv::Mat guide, cv::Mat scale, float sigma_d, float sigma_r);

	cv::Mat1i InteractiveAdjScale(cv::Mat1i& adjScale, cv::Mat1i  mask, cv::Mat1i  fix_mask);
	cv::Mat1i ConditionAdjScale(cv::Mat1i& adjScale, cv::Mat1i  mask);
	cv::Mat1i ReviseScale(cv::Mat1i& adjScale);

	void DetailEnhancement(const cv::Mat input, const cv::Mat flt, int sigma, int l_size, int iter);
};
