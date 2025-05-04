#include "LineFiltering.h"
#include<cmath>
#include<algorithm>

using namespace std;

LineFiltering::LineFiltering()
{
}

LineFiltering::~LineFiltering()
{
}

cv::Mat LineFiltering::BRG2Y(cv::Mat img)
{
	return cv::Mat();
}

void LineFiltering::GetMLD(const cv::Mat & input, cv::Mat & mld, cv::Mat1i& maxDir, int pSize)
{
	//Maximum Local Difference
	int height = input.rows;
	int width = input.cols;
	// 先根据pSize求均值
	cv::Mat  mean_filter = MeanFiltering(input, pSize);
	mld = cv::Mat::zeros(height, width, CV_32FC3);
	maxDir = cv::Mat1i::zeros(height, width);
	int dir[4][2] = { {1,1},{0,1},{-1,1},{-1,0} };    //direction
	for (int x = pSize; x < height - pSize; x++)
	{
		for (int y = pSize; y < width - pSize; y++)
		{
			float diff_Value = 0.0f;
			int maxdir_Value = 0;
			cv::Vec2i f_pnt(0, 0);
			cv::Vec2i b_pnt(0, 0);
			cv::Vec3f diff_Vec(0.0f, 0.0f, 0.0f);
			for (int k = 0; k < 4; k++)
			{
				int x_f = x + pSize * dir[k][0];
				int y_f = y + pSize * dir[k][1];
				int x_b = x - pSize * dir[k][0];
				int y_b = y - pSize * dir[k][1];
				cv::Vec3f f_value = mean_filter.at<cv::Vec3f>(x_f, y_f);
				cv::Vec3f b_value = mean_filter.at<cv::Vec3f>(x_b, y_b);
				float value = (f_value - b_value).dot(f_value - b_value);
				if (value > diff_Value)
				{
					diff_Value = value;
					diff_Vec = f_value - b_value;
					maxdir_Value = dir[k][0] + dir[k][1];
				}
			}
			mld.at<cv::Vec3f>(x, y) = diff_Vec;
			maxDir.at<int>(x, y) = maxdir_Value;
		}
	}

}

void LineFiltering::GetMLDwith2x2(const cv::Mat & input, cv::Mat & mld, cv::Mat1i & maxDir)
{
	//Maximum Local Difference
	int height = input.rows;
	int width = input.cols;
	cv::Mat sum;
	integral(input, sum);//计算积分图像及平方像素值积分图像
	mld = cv::Mat::zeros(height, width, CV_32FC3);
	maxDir = cv::Mat1i::zeros(height, width);
	int dir[4][2] = { {1,1},{0,1},{-1,1},{-1,0} };    //direction
	int pSize = 1;


	Mat f_and_b(4, 2, CV_32FC3); //建立一个三行三列3通道像素
	for (int x = pSize; x < height - pSize; x++)
	{
		for (int y = pSize; y < width - pSize; y++)
		{
			cv::Vec3f f_value1 = sum.at<Vec3d>(x + 1 + 1, y + 1 + 1) - sum.at<Vec3d>(x, y + 1 + 1) - sum.at<Vec3d>(x + 1 + 1, y) + sum.at<Vec3d>(x, y);
			cv::Vec3f b_value1 = sum.at<Vec3d>(x + 1, y + 1) - sum.at<Vec3d>(x - 1, y + 1) - sum.at<Vec3d>(x + 1, y - 1) + sum.at<Vec3d>(x - 1, y - 1);
			f_and_b.at< Vec3f>(0, 0) = f_value1 / 4;
			f_and_b.at< Vec3f>(0, 1) = b_value1 / 4;

			cv::Vec3f f_value2 = sum.at<Vec3d>(x + 1 + 1, y + 1 + 1) - sum.at<Vec3d>(x, y + 1 + 1) - sum.at<Vec3d>(x + 1 + 1, y - 1) + sum.at<Vec3d>(x, y - 1);
			cv::Vec3f b_value2 = sum.at<Vec3d>(x + 1, y + 1 + 1) - sum.at<Vec3d>(x - 1, y + 1 + 1) - sum.at<Vec3d>(x + 1, y - 1) + sum.at<Vec3d>(x - 1, y - 1);
			f_and_b.at< Vec3f>(1, 0) = f_value2 / 6;
			f_and_b.at< Vec3f>(1, 1) = b_value2 / 6;

			cv::Vec3f f_value3 = sum.at<Vec3d>(x + 1 + 1, y + 1) - sum.at<Vec3d>(x, y + 1) - sum.at<Vec3d>(x + 1 + 1, y - 1) + sum.at<Vec3d>(x, y - 1);
			cv::Vec3f b_value3 = sum.at<Vec3d>(x + 1, y + 1 + 1) - sum.at<Vec3d>(x - 1, y + 1 + 1) - sum.at<Vec3d>(x + 1, y) + sum.at<Vec3d>(x - 1, y);
			f_and_b.at< Vec3f>(2, 0) = f_value3 / 4;
			f_and_b.at< Vec3f>(2, 1) = b_value3 / 4;

			cv::Vec3f f_value4 = sum.at<Vec3d>(x + 1 + 1, y + 1 + 1) - sum.at<Vec3d>(x - 1, y + 1 + 1) - sum.at<Vec3d>(x + 1 + 1, y) + sum.at<Vec3d>(x - 1, y);
			cv::Vec3f b_value4 = sum.at<Vec3d>(x + 1 + 1, y + 1) - sum.at<Vec3d>(x - 1, y + 1) - sum.at<Vec3d>(x + 1 + 1, y - 1) + sum.at<Vec3d>(x - 1, y - 1);
			f_and_b.at< Vec3f>(3, 0) = f_value4 / 6;
			f_and_b.at< Vec3f>(3, 1) = b_value4 / 6;

			float diff_Value = 0.0f;
			int maxdir_Value = 0;
			cv::Vec3f diff_Vec(0.0f, 0.0f, 0.0f);
			for (int k = 0; k < 4; k++)
			{
				cv::Vec3f f_value = f_and_b.at<cv::Vec3f>(k, 0);
				cv::Vec3f b_value = f_and_b.at<cv::Vec3f>(k, 1);
				float value = (f_value - b_value).dot(f_value - b_value);
				if (value > diff_Value)
				{
					diff_Value = value;
					diff_Vec = f_value - b_value;
					maxdir_Value = dir[k][0] + dir[k][1];
				}
			}
			mld.at<cv::Vec3f>(x, y) = diff_Vec;
			maxDir.at<int>(x, y) = maxdir_Value;
		}

		int  a = 0;
	}
}

cv::Mat LineFiltering::MeanFiltering(const cv::Mat & input, int pSize)
{
	//对每个像素点，求 pSize 的patch 的均值，以实现平滑
	int height = input.rows;
	int width = input.cols;
	cv::Mat filter_result = cv::Mat::zeros(height, width, CV_32FC3);
	int x_min, x_max, y_min, y_max;
	for (int x = 0; x < height; x++)
	{
		for (int y = 0; y < width; y++)
		{
			x_min = max(0, x - pSize);
			x_max = min(height - 1, x + pSize);
			y_min = max(0, y - pSize);
			y_max = min(width - 1, y + pSize);
			cv::Vec3f mean_Value(0.0f, 0.0f, 0.0f);
			int num_pixels = (x_max - x_min + 1) * (y_max - y_min + 1);
			for (int xx = x_min; xx <= x_max; xx++)
			{
				for (int yy = y_min; yy <= y_max; yy++)
				{
					mean_Value += input.at<Vec3f>(xx, yy);
				}
			}
			filter_result.at<cv::Vec3f>(x, y) = mean_Value / num_pixels;
		}
	}
	return filter_result;
}

cv::Mat LineFiltering::nonlinearEnhance(const cv::Mat & mld, int pSize, int sigma)
{
	// Nonlinear Enhancement of Our MLD
	int height = mld.rows;
	int width = mld.cols;
	cv::Mat enhanceMP = cv::Mat::zeros(mld.rows, mld.cols, CV_32F);
	for (int x = 0; x < height; x++)
	{
		for (int y = 0; y < width; y++)
		{
			cv::Vec3f  vec_t = mld.at<cv::Vec3f>(x, y);
			float t_Value = vec_t.dot(vec_t);
			enhanceMP.at<float>(x, y) = 2.0 * (1.0 / (1 + exp(-1 * sigma *(2 * pSize + 1) * (t_Value))) - 0.5);
		}
	}
	return enhanceMP;
}

cv::Mat LineFiltering::heuristicMP(const cv::Mat input, int t)
{
	int m_w = input.cols;
	int m_h = input.rows;
	cv::Mat1f re(input.rows, input.cols);
	int x_min, x_max, y_min, y_max;
	for (int y = 0; y < input.rows; y++)
	{
		for (int x = 0; x < input.cols; x++)
		{
			x_min = max(0, x - t);
			x_max = min(m_w - 1, x + t);
			y_min = max(0, y - t);
			y_max = min(m_h - 1, y + t);
			float temp = 0.0f;
			int count = (y_max - y_min + 1) * (y_max - y_min + 1);
			int c_Ok = 0;
			for (int yy = y_min; yy <= y_max; yy++)
			{
				for (int xx = x_min; xx <= x_max; xx++)
				{
					temp = max(temp, input.at<float>(yy, xx));
					if (input.at<float>(yy, xx) > 0.5)
					{
						c_Ok++;
					}
				}
			}
			if (c_Ok >= 1.0 / t * count)
				re.at<float>(y, x) = temp;
			else
				re.at<float>(y, x) = input.at<float>(y, x);
		}
	}
	return re;
}

cv::Mat1i LineFiltering::InitialScale(const cv::Mat enhanceMP, int t, int minscale, bool isfixminS)
{
	int height = enhanceMP.rows;
	int width = enhanceMP.cols;
	cv::Mat1i scale_img(height, width);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			float alpha = enhanceMP.at<float>(i, j);
			if (isfixminS)
			{
				scale_img.at<int>(i, j) = max(int((1.0 - alpha) * t + 0.5), minscale);      //原徐盼盼的处理方式，最小窗口为3*3
			}
			else
			{
				scale_img.at<int>(i, j) = int((1.0 - alpha) * t + 0.5);       //该公式窗口可能会出现1*1，即原像素不变，不被滤除。
			}
		}
	}
	return scale_img;
}

cv::Mat1i LineFiltering::AdjustScale(cv::Mat1i & initialScale, cv::Mat & inputImg, int pSize, const cv::Mat& mld, cv::Mat& final_dir)
{
	//只对窗口为3*3的处理
	int height = inputImg.rows;
	int width = inputImg.cols;
	cv::Mat temp_mld;
	final_dir = cv::Mat1i::zeros(height, width);
	//如果pSize = 1，则已用patch size = 2*2（对应对角线上的两个方向） 或2*3（对应上下左右两个方向）的窗口进行了均值平滑 
	if (pSize == 1)
	{
		cv::Mat mld2x2;
		cv::Mat1i maxDir2x2;
		GetMLDwith2x2(inputImg, mld2x2, maxDir2x2);
		temp_mld = mld2x2.clone();
		final_dir = maxDir2x2.clone();
	}
	else   //如果pSize > 1，则用patch size = 3的窗口进行了均值平滑 
	{
		pSize = 1;   //对应3*3的窗口
		cv::Mat mean_filter = MeanFiltering(inputImg, pSize);
		//对原图再进行MLD   using 3*3 pixels
		cv::Mat mld3x3;
		cv::Mat1i maxDir;          // store the direction of the max MLD
		GetMLD(inputImg, mld3x3, maxDir, pSize);
		temp_mld = mld3x3.clone();
		final_dir = maxDir.clone();
	}

	cv::Mat1i adjustScale;
	initialScale.copyTo(adjustScale);
	float alph = 0.9;                          //调整新MLD 和旧MLD的大小比例
	for (int x = pSize; x < height - pSize; x++)
	{
		for (int y = pSize; y < width - pSize; y++)
		{
			//与原patch对应的mld进行比较
			cv::Vec3f  vec_new = temp_mld.at<cv::Vec3f>(x, y);
			float new_Value = vec_new.dot(vec_new);
			cv::Vec3f  vec_old = mld.at<cv::Vec3f>(x, y);
			float old_Value = vec_old.dot(vec_old);

			bool isEdge = (new_Value > alph*old_Value);
			int scale_pixle = initialScale(x, y);
			if (scale_pixle == 0)
			{
				if (isEdge)
				{
					adjustScale(x, y) = 0;
				}
				else
				{
					adjustScale(x, y) = 1;
				}
			}
		}
	}	return adjustScale;
}

cv::Mat LineFiltering::GetGuidance(const cv::Mat input, const cv::Mat1i scale, const cv::Mat1i& final_dir)
{
	int width = input.cols;
	int height = input.rows;
	cv::Mat3f resultG = cv::Mat3f::zeros(height, width);
	int x_min, x_max, y_min, y_max;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int k = scale.at<int>(i, j);
			if (k > 0 || j == 0 || j == width - 1 || i == 0 || i == height - 1)
			{
				x_min = max(0, j - k);
				x_max = min(width - 1, j + k);
				y_min = max(0, i - k);
				y_max = min(height - 1, i + k);
				cv::Vec3f temp(0.0, 0.0, 0.0);
				int count = (y_max - y_min + 1) * (x_max - x_min + 1);
				for (int y = y_min; y <= y_max; y++)
				{
					for (int x = x_min; x <= x_max; x++)
					{
						cv::Vec3f p = input.at<cv::Vec3f>(y, x);
						temp += p;
					}
				}
				temp /= count;
				resultG.at<cv::Vec3f>(i, j) = temp;
			}
			else   //k=0
			{
				int dir = final_dir.at<int>(i, j);
				cv::Vec3f temp(0.0, 0.0, 0.0);
				switch (dir)
				{
				case 2:
					temp = input.at<cv::Vec3f>(i - 1, j + 1) + input.at<cv::Vec3f>(i, j) + input.at<cv::Vec3f>(i + 1, j - 1);
					resultG.at<cv::Vec3f>(i, j) = temp / 3;
					break;
				case 1:
					temp = input.at<cv::Vec3f>(i, j + 1) + input.at<cv::Vec3f>(i, j) + input.at<cv::Vec3f>(i, j - 1);
					resultG.at<cv::Vec3f>(i, j) = temp / 3;
					break;
				case 0:
					temp = input.at<cv::Vec3f>(i - 1, j - 1) + input.at<cv::Vec3f>(i, j) + input.at<cv::Vec3f>(i + 1, j + 1);
					resultG.at<cv::Vec3f>(i, j) = temp / 3;
					break;
				case -1:
					temp = input.at<cv::Vec3f>(i - 1, j) + input.at<cv::Vec3f>(i, j) + input.at<cv::Vec3f>(i + 1, j);
					resultG.at<cv::Vec3f>(i, j) = temp / 3;
					break;
				default:
					cout << "error!" << endl;
					break;
				}
			}
		}
	}
	return resultG;
}

cv::Mat LineFiltering::lfHtfilter_o(cv::Mat m_img3f, cv::Mat guide, int psize, float sigma_d, float sigma_r)
{
	//一维滤波
	int m_h = m_img3f.rows;
	int m_w = m_img3f.cols;

	Mat A = m_img3f;

	Mat B, mG;
	mG = guide;
	cvtColor(m_img3f, A, COLOR_BGR2Lab);
	//cvtColor(guide,mG,CV_BGR2Lab);
	B = A;
	sigma_r = sigma_r;
	//sigma_r = 100*sigma_r;
	Mat gauss(2 * psize + 1, 2 * psize + 1, CV_32FC1);
	Mat lf_gauss(1, 2 * psize + 1, CV_32FC1);
	Mat temp_gauss = Mat1f::zeros(1, 2 * psize + 1);
	float temp = 0;

	for (int x = -psize; x <= psize; x++)
	{
		lf_gauss.at<float>(0, psize + x) = exp(-(x * x) / (2.0 * sigma_d * sigma_d));

		temp = temp + lf_gauss.at<float>(0, psize + x);
	}
	temp_gauss = lf_gauss / temp;

	int x_min, x_max, y_min, y_max;

	for (int y = 0; y < m_h; y++)
		for (int x = 0; x < m_w; x++)
		{
			int k = psize;
			x_min = max(0, x - k);
			x_max = min(m_w - 1, x + k);
			y_min = max(0, y - k);
			y_max = min(m_h - 1, y + k);

			float tot_f = 0.0f;
			Vec3f bfc(0.0f, 0.0f, 0.0f);

			for (int xx = x_min; xx <= x_max; xx++)   //横向滤波
			{
				Vec3f IA = mG.at<Vec3f>(y, xx);
				Vec3f dif = IA - mG.at<Vec3f>(y, x);
				float temp = exp(-(dif[0] * dif[0] + dif[1] * dif[1] + dif[2] * dif[2]) / (2.0 * sigma_r * sigma_r));
				float ftemp = temp * lf_gauss.at<float>(0, xx - x + k);
				Vec3f JA = A.at<Vec3f>(y, xx);
				bfc[0] += ftemp * JA[0]; bfc[1] += ftemp * JA[1]; bfc[2] += ftemp * JA[2];
				tot_f += ftemp;
			}
			bfc /= tot_f;
			B.at<Vec3f>(y, x) = bfc;
		}

	cvtColor(B, A, COLOR_Lab2BGR);
	return A;
}

cv::Mat LineFiltering::lfVtfilter_o(cv::Mat m_img3f, cv::Mat guide, int psize, float sigma_d, float sigma_r)
{
	//一维滤波
	int m_h = m_img3f.rows;
	int m_w = m_img3f.cols;

	Mat A = m_img3f;

	Mat B, mG;
	mG = guide;
	cvtColor(m_img3f, A, COLOR_BGR2Lab);
	//cvtColor(guide,mG,CV_BGR2Lab);
	B = A;
	sigma_r = sigma_r;
	//sigma_r = 100*sigma_r;
	Mat gauss(2 * psize + 1, 2 * psize + 1, CV_32FC1);
	Mat lf_gauss(1, 2 * psize + 1, CV_32FC1);
	Mat temp_gauss = Mat1f::zeros(1, 2 * psize + 1);
	float temp = 0;

	for (int x = -psize; x <= psize; x++)
	{
		lf_gauss.at<float>(0, psize + x) = exp(-(x * x) / (2.0 * sigma_d * sigma_d));

		temp = temp + lf_gauss.at<float>(0, psize + x);
	}
	temp_gauss = lf_gauss / temp;
	lf_gauss = lf_gauss.t();

	int x_min, x_max, y_min, y_max;

	for (int x = 0; x < m_w; x++)
		for (int y = 0; y < m_h; y++)
		{
			//int k = scale.at<int>(y,x);
			int k = psize;
			//x_min = max(0, x - k);
			//x_max = min(m_w - 1, x + k);
			y_min = max(0, y - k);
			y_max = min(m_h - 1, y + k);

			float tot_f = 0.0f;
			Vec3f bfc(0.0f, 0.0f, 0.0f);

			for (int yy = y_min; yy <= y_max; yy++)   //横向滤波
			{
				Vec3f IA = mG.at<Vec3f>(yy, x);
				Vec3f dif = IA - mG.at<Vec3f>(y, x);
				float temp = exp(-(dif[0] * dif[0] + dif[1] * dif[1] + dif[2] * dif[2]) / (2.0 * sigma_r * sigma_r));
				float ftemp = temp * lf_gauss.at<float>(yy - y + k, 0);
				Vec3f JA = A.at<Vec3f>(yy, x);
				bfc[0] += ftemp * JA[0]; bfc[1] += ftemp * JA[1]; bfc[2] += ftemp * JA[2];
				tot_f += ftemp;
			}
			bfc /= tot_f;
			B.at<Vec3f>(y, x) = bfc;
		}

	cvtColor(B, A, COLOR_Lab2BGR);
	return A;
}

cv::Mat LineFiltering::LFHfilter_VariableScale(cv::Mat input, cv::Mat guide, cv::Mat scale, float sigma_d, float sigma_r)
{
	int m_h = input.rows;
	int m_w = input.cols;

	Mat A = input;
	Mat B, mG;
	mG = guide;
	cvtColor(input, A, COLOR_BGR2Lab);
	B = A;
	int x_min, x_max, y_min, y_max;
	for (int y = 0; y < m_h; y++)
	{
		for (int x = 0; x < m_w; x++)
		{
			int kernal_size = scale.at<int>(y, x);
			if (kernal_size == 0)
				kernal_size = 1;

			//kernal_size = kernal_size + 1;

			x_min = max(0, x - kernal_size);
			x_max = min(m_w - 1, x + kernal_size);
			//y_min = max(0, y - kernal_size);
			//y_max = min(m_h - 1, y + kernal_size);

			Mat lf_gauss(1, 2 * kernal_size + 1, CV_32FC1);
			for (int x = -kernal_size; x <= kernal_size; x++)
			{
				lf_gauss.at<float>(0, kernal_size + x) = exp(-(x * x) / (2.0 * sigma_d * sigma_d));
			}

			float tot_f = 0.0f;
			Vec3f bfc(0.0f, 0.0f, 0.0f);
			for (int xx = x_min; xx <= x_max; xx++)   //横向滤波
			{
				Vec3f IA = mG.at<Vec3f>(y, xx);
				Vec3f dif = IA - mG.at<Vec3f>(y, x);
				float temp = exp(-(dif[0] * dif[0] + dif[1] * dif[1] + dif[2] * dif[2]) / (2.0 * sigma_r * sigma_r));
				float ftemp = temp * lf_gauss.at<float>(0, xx - x + kernal_size);
				Vec3f JA = A.at<Vec3f>(y, xx);
				bfc[0] += ftemp * JA[0]; bfc[1] += ftemp * JA[1]; bfc[2] += ftemp * JA[2];
				tot_f += ftemp;
			}
			bfc /= tot_f;
			B.at<Vec3f>(y, x) = bfc;
		}
	}
	cvtColor(B, A, COLOR_Lab2BGR);
	return A;
}

cv::Mat LineFiltering::LFVfilter_VariableScale(cv::Mat input, cv::Mat guide, cv::Mat scale, float sigma_d, float sigma_r)
{
	int m_h = input.rows;
	int m_w = input.cols;
	Mat A = input;
	Mat B, mG;
	mG = guide;
	cvtColor(input, A, COLOR_BGR2Lab);
	B = A;

	int x_min, x_max, y_min, y_max;
	for (int x = 0; x < m_w; x++)
	{
		for (int y = 0; y < m_h; y++)
		{
			int kernal_size = scale.at<int>(y, x);
			if (kernal_size == 0)
				kernal_size = 1;
			//kernal_size = kernal_size + 1;

			y_min = max(0, y - kernal_size);
			y_max = min(m_h - 1, y + kernal_size);

			Mat lf_gauss(1, 2 * kernal_size + 1, CV_32FC1);

			for (int x = -kernal_size; x <= kernal_size; x++)
			{
				lf_gauss.at<float>(0, kernal_size + x) = exp(-(x * x) / (2.0 * sigma_d * sigma_d));
			}
			lf_gauss = lf_gauss.t();

			float tot_f = 0.0f;
			Vec3f bfc(0.0f, 0.0f, 0.0f);

			for (int yy = y_min; yy <= y_max; yy++)   //横向滤波
			{
				Vec3f IA = mG.at<Vec3f>(yy, x);
				Vec3f dif = IA - mG.at<Vec3f>(y, x);
				float temp = exp(-(dif[0] * dif[0] + dif[1] * dif[1] + dif[2] * dif[2]) / (2.0 * sigma_r * sigma_r));
				float ftemp = temp * lf_gauss.at<float>(yy - y + kernal_size, 0);
				Vec3f JA = A.at<Vec3f>(yy, x);
				bfc[0] += ftemp * JA[0]; bfc[1] += ftemp * JA[1]; bfc[2] += ftemp * JA[2];
				tot_f += ftemp;
			}
			bfc /= tot_f;
			B.at<Vec3f>(y, x) = bfc;
		}
	}
	cvtColor(B, A, COLOR_Lab2BGR);
	return A;
}

cv::Mat LineFiltering::bftfilter_o(cv::Mat m_img3f, cv::Mat guide, int ksize, float sigma_d, float sigma_r)
{
	int m_h = m_img3f.rows;
	int m_w = m_img3f.cols;

	Mat A = m_img3f;

	Mat B, mG;
	mG = guide;
	cvtColor(m_img3f, A, COLOR_BGR2Lab);
	//cvtColor(guide,mG,CV_BGR2Lab);
	B = A;
	sigma_r = sigma_r;
	//sigma_r = 100*sigma_r;
	Mat gauss(2 * ksize + 1, 2 * ksize + 1, CV_32FC1);
	Mat temp_gauss = Mat1f::zeros(2 * ksize + 1, 2 * ksize + 1);
	float temp = 0;
	for (int y = -ksize; y <= ksize; y++)
	{
		for (int x = -ksize; x <= ksize; x++)
		{
			gauss.at<float>(ksize + y, ksize + x) = exp(-(y * y + x * x) / (2.0 * sigma_d * sigma_d));

			temp = temp + gauss.at<float>(ksize + y, ksize + x);
		}
	}
	temp_gauss = gauss / temp;

	int x_min, x_max, y_min, y_max;

	for (int y = 0; y < m_h; y++)
		for (int x = 0; x < m_w; x++)
		{
			//int k = scale.at<int>(y,x);
			int k = ksize;
			x_min = max(0, x - k);
			x_max = min(m_w - 1, x + k);
			y_min = max(0, y - k);
			y_max = min(m_h - 1, y + k);
			int y_h = (y_max - y_min) + 1;
			int x_w = (x_max - x_min) + 1;
			//Mat F(y_h,x_w,CV_32FC1);
			float tot_f = 0.0f;
			Vec3f bfc(0.0f, 0.0f, 0.0f);

			for (int yy = y_min; yy <= y_max; yy++)
				for (int xx = x_min; xx <= x_max; xx++)
				{
					Vec3f IA = mG.at<Vec3f>(yy, xx);
					Vec3f JA = A.at<Vec3f>(yy, xx);
					Vec3f dif = IA - mG.at<Vec3f>(y, x);
					float temp = exp(-(dif[0] * dif[0] + dif[1] * dif[1] + dif[2] * dif[2]) / (2.0 * sigma_r * sigma_r));
					float ftemp = temp * gauss.at<float>(yy - y + k, xx - x + k);
					bfc[0] += ftemp * JA[0]; bfc[1] += ftemp * JA[1]; bfc[2] += ftemp * JA[2];
					tot_f += ftemp;
					//F.at<float>(yy-y_min,xx-x_min) = ftemp;
				}
			bfc /= tot_f;
			B.at<Vec3f>(y, x) = bfc;
		}

	cvtColor(B, A, COLOR_Lab2BGR);
	return A;
}

cv::Mat LineFiltering::bftfilter_varyingwindow(cv::Mat input, cv::Mat guide, cv::Mat scale, float sigma_d, float sigma_r)
{
	int m_h = input.rows;
	int m_w = input.cols;
	Mat A = input;

	Mat B, mG;
	mG = guide;
	cvtColor(input, A, COLOR_BGR2Lab);
	B = A;

	int x_min, x_max, y_min, y_max;
	for (int x = 0; x < m_w; x++)
	{
		for (int y = 0; y < m_h; y++)
		{
			int kernal_size = scale.at<int>(y, x);
			if (kernal_size == 0)
				kernal_size = 1;

			//kernal_size = kernal_size + 1;

			Mat gauss(2 * kernal_size + 1, 2 * kernal_size + 1, CV_32FC1);
			for (int y = -kernal_size; y <= kernal_size; y++)
			{
				for (int x = -kernal_size; x <= kernal_size; x++)
				{
					gauss.at<float>(kernal_size + y, kernal_size + x) = exp(-(y*y + x * x) / (2.0*sigma_d*sigma_d));
				}
			}

			x_min = max(0, x - kernal_size);
			x_max = min(m_w - 1, x + kernal_size);
			y_min = max(0, y - kernal_size);
			y_max = min(m_h - 1, y + kernal_size);

			float tot_f = 0.0f;
			Vec3f bfc(0.0f, 0.0f, 0.0f);

			for (int yy = y_min; yy <= y_max; yy++)
			{
				for (int xx = x_min; xx <= x_max; xx++)
				{
					Vec3f IA = mG.at<Vec3f>(yy, xx);
					Vec3f JA = A.at<Vec3f>(yy, xx);
					Vec3f dif = IA - mG.at<Vec3f>(y, x);
					float temp = exp(-(dif[0] * dif[0] + dif[1] * dif[1] + dif[2] * dif[2]) / (2.0*sigma_r*sigma_r));
					float ftemp = temp * gauss.at<float>(yy - y + kernal_size, xx - x + kernal_size);
					bfc[0] += ftemp * JA[0]; bfc[1] += ftemp * JA[1]; bfc[2] += ftemp * JA[2];
					tot_f += ftemp;
				}
			}
			bfc /= tot_f;
			B.at<Vec3f>(y, x) = bfc;
		}
	}
	cvtColor(B, A, COLOR_Lab2BGR);
	return A;
}

cv::Mat1i LineFiltering::InteractiveAdjScale(cv::Mat1i & adjScale, cv::Mat1i mask, cv::Mat1i  fix_mask)
{
	int height = adjScale.rows;
	int width = adjScale.cols;
	int tempScale = 0;
	int fixScale = 0;
	cv::Mat1i adjustScale;
	adjScale.copyTo(adjustScale);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			tempScale = mask.at<int>(i, j);
			fixScale = fix_mask.at<int>(i, j);
			if (tempScale != 0 || fixScale != 0)
			{
				if (tempScale == 0)
				{
					adjustScale.at<int>(i, j) = fixScale;
				}
				else
				{
					adjustScale.at<int>(i, j) = tempScale;
				}
			}
		}
	}
	return adjustScale;
}

cv::Mat1i LineFiltering::ConditionAdjScale(cv::Mat1i & adjScale, cv::Mat1i mask)
{
	int height = adjScale.rows;
	int width = adjScale.cols;
	int tempScale = 0;
	int orgScale = -1;
	cv::Mat1i adjustScale;
	adjScale.copyTo(adjustScale);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			orgScale = adjScale.at<int>(i, j);
			tempScale = mask.at<int>(i, j);
			if (tempScale != 0 && orgScale == 0)
			{
				adjustScale.at<int>(i, j) = tempScale;
			}
		}
	}
	return adjustScale;
}


void LineFiltering::DetailEnhancement(const cv::Mat input, const cv::Mat flt, int sigma, int l_size, int iter)
{
	for (int i = 1; i < 5; i++)
	{


		int alph = i;
		int width = input.cols;
		int height = input.rows;
		cv::Mat3f DiffMap = cv::Mat3f::zeros(height, width);
		cv::Mat3f detailEImg = cv::Mat3f::zeros(height, width);
		int x_min, x_max, y_min, y_max;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				cv::Vec3f  vec_input = input.at<cv::Vec3f>(i, j);
				cv::Vec3f  vec_flt = flt.at<cv::Vec3f>(i, j);
				cv::Vec3f diff = vec_input - vec_flt;

				DiffMap.at<cv::Vec3f>(i, j) = diff;

				detailEImg.at<cv::Vec3f>(i, j) = vec_input + diff * alph;
			}
		}
		String strName1 = "supple_materials/our_results/fig52/detailEnhanceInter/";
		String strName2 = ".png";
		String strName3 = "_HV_";
		String strName4 = "alph_";
		String strName6 = "sigma_FixS";

		String strName5 = strName1 + to_string(2 * l_size + 1) + strName3 + to_string(alph) + strName4 + to_string(sigma) + strName6 + to_string(iter) + strName2;
		cv::imwrite(strName5, detailEImg*255.0);
	}

}

