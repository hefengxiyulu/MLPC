// LineFiltering.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include <iostream>
#include<opencv2/opencv.hpp>
#include<Windows.h>
#include<stdio.h>
#include<string>
#include"LineFiltering.h"
#include"TextureFiltering.h"

using namespace std;
using namespace cv;

#define WINDOW_NAME "draw"

//全局函数声明部分
void on_MouseHandle(int event, int x, int y, int flags, void* param);
void DrawRectangle(cv::Mat& img, cv::Rect box);  //在临时变量的图片上绘制矩形
void ShowHelpText();
RNG g_rng(12345);
bool lButtonDown = false;
int scribbleRadius = 5;
int g_Scale = 1;
Mat showImg;
Mat fgScribbleMask;

int main()
{
	// 统计时间的定义
	LARGE_INTEGER _freq;
	LARGE_INTEGER _begin;
	LARGE_INTEGER _end;
	QueryPerformanceFrequency(&_freq);

	/*------------导入图像------------------------*/
	cv::Mat3b input = cv::imread("supple_materials/png_img/000007.png");
	cv::Mat3f img, tempImg;
	input.convertTo(img, CV_32FC3, 1.0 / 255);
	input.convertTo(tempImg, CV_32FC3, 1.0 / 255);

	int sigma = 2;                //  𝜎𝑠, Eq.2的参数设置
	img = tempImg.clone();

	/*------------参数设置------------------------*/
	LineFiltering lf;              //创建对象
	TextureFiltering tf;
	int n_iter = 6;                 //迭代次数N
	int l_size = 6;                //初始滤波尺度t
	int k_size = l_size;

	cv::Mat flt(img.size(), CV_32FC3, Scalar(0, 0, 0));            //存放迭代滤波的结果
	cv::Mat1i fix_mask = cv::Mat1i::zeros(flt.rows, flt.cols);

	bool isHV_fixS = true;
	bool isHV_VaryS = false;
	bool isWind_fixS = false;
	bool isWind_varyS = false;
	bool isInter = false;
	bool isfixminS = false;

	QueryPerformanceCounter(&_begin);       //获得起始值  

	/*------------迭代平滑------------------------*/
	for (int i = 0; i < n_iter; i++)
	{
		if (isWind_varyS == true)
		{
			if (i > 1)
				k_size = k_size - 1;
		}

		//compute MLPC
		cv::Mat mld;                    //store  the value of MLPC
		cv::Mat1i maxDir;          // store the direction of the max MLPC
		cv::Mat1i final_Dir;       //final direction of the max MLPC
		int p_size = max((l_size * 2 + 1) / 6, 1);

		//QueryPerformanceCounter(&_begin);       //获得起始值  

		lf.GetMLD(img, mld, maxDir, p_size);                                                            // Eq. 1     Maximum Local Difference
		cv::Mat MP = lf.nonlinearEnhance(mld, l_size + 1, sigma);                            // Eq.2  Nonlinear Enhancement of Our MLPC

		//QueryPerformanceCounter(&_end);      //获得终止值      
		//float costTime = (float((_end.QuadPart - _begin.QuadPart) / 1.0 / _freq.QuadPart));
		//printf("MLD & MP cost time emld: %.5f\n", costTime);

		//QueryPerformanceCounter(&_begin);       //获得起始值  
		cv::Mat heuristicMP = lf.heuristicMP(MP, (l_size + 1) / 2);                 // Eq.4   heuristic statistic-based measure
		cv::Mat1i initialscale = lf.InitialScale(heuristicMP, k_size, 1, isfixminS);            // 计算每个像素处的平滑尺度
		//cv::Mat1i initialscale = lf.InitialScale(MP, k_size, 1);                                         // 计算每个像素处的平滑尺度
		cv::Mat1i  adjustScale = lf.AdjustScale(initialscale, img, p_size, mld, final_Dir);

		//QueryPerformanceCounter(&_end);      //获得终止值      
		//costTime = (float((_end.QuadPart - _begin.QuadPart) / 1.0 / _freq.QuadPart)); //计算耗时
		//printf("cost time k G: %.5f\n", costTime);

		cv::Mat1i  inter_adjustScale;
		if (isInter == true)
		{
			cv::Mat1i mask = cv::Mat1i::zeros(flt.rows, flt.cols);
			fgScribbleMask.create(2, flt.size, CV_8UC1);
			fgScribbleMask = 0;
			showImg = img.clone();
			Mat srcImage, tempImage;
			srcImage = img.clone();

			namedWindow(WINDOW_NAME, CV_WINDOW_NORMAL);
			createTrackbar("thick", "draw", &scribbleRadius, 10, 0);
			createTrackbar("Scale", "draw", &g_Scale, 10, 0);
			setMouseCallback(WINDOW_NAME, on_MouseHandle, (void*)&srcImage);      //设置鼠标操作回调函数
			while (1)
			{//srcImage.copyTo(tempImage); //复制源图到临时变量
				if (waitKey(10) == 27)
				{
					mask = fgScribbleMask.clone();
					break;
				}
			}
			if (i == 0)
				fix_mask = mask;

			inter_adjustScale = lf.InteractiveAdjScale(adjustScale, mask, fix_mask);
			//cv::Mat1i  inter_adjustScale = lf.ConditionAdjScale(adjustScale, mask);
		}

		//QueryPerformanceCounter(&_begin);       //获得起始值  
		cv::Mat3f Guidance;
		if (isInter == true)
		{
			Guidance = lf.GetGuidance(img, inter_adjustScale, final_Dir);
		}
		else
		{
			Guidance = lf.GetGuidance(img, adjustScale, final_Dir);
		}

		if (isHV_fixS == true)
		{
			// fixed sacle l_size
			flt = lf.lfHtfilter_o(img, Guidance, l_size, l_size, 0.05 * sqrt(3));
			flt = lf.lfVtfilter_o(flt, Guidance, l_size, l_size, 0.05 * sqrt(3));
		}


		if (isHV_VaryS == true)
		{
			// Variable scale
			flt = lf.LFHfilter_VariableScale(img, Guidance, adjustScale, l_size, 0.05 * sqrt(3));
			flt = lf.LFVfilter_VariableScale(flt, Guidance, adjustScale, l_size, 0.05 * sqrt(3));
		}

		img = flt;
		String strName2 = ".png";
		String strName3 = "supple_materials/our_results/fig23/";
		String strName5 = "_HV_";
		String strName6;
		if (isWind_varyS)    //初始化计算G的窗口是否随着迭代逐渐减小
		{
			strName6 = "sigma_varyWFixS";
		}
		else
		{
			strName6 = "sigma_FixS";
		}
		String strName7 = "sigma_minWindFixS";
		String strName8 = "_HV_inter_";
		String strName4;
		if (isfixminS)      //窗口是否可以为0
		{
			if (isInter)
				strName4 = strName3 + to_string(2 * l_size + 1) + strName8 + to_string(sigma) + strName7 + to_string(i) + strName2;
			else
				strName4 = strName3 + to_string(2 * l_size + 1) + strName5 + to_string(sigma) + strName7 + to_string(i) + strName2;
		}
		else
		{
			if (isInter)
				strName4 = strName3 + to_string(2 * l_size + 1) + strName8 + to_string(sigma) + strName6 + to_string(i) + strName2;
			else
				strName4 = strName3 + to_string(2 * l_size + 1) + strName5 + to_string(sigma) + strName6 + to_string(i) + strName2;

		}

		if (i >= 0)
			cv::imwrite(strName4, flt*255.0);

			if (isInter)
			{
				cv::destroyWindow(WINDOW_NAME);
				//cv::destroyWindow("fg mask");
			}
	}

	QueryPerformanceCounter(&_end);                           //获得终止值      
	float costTime = (float((_end.QuadPart - _begin.QuadPart) / 1.0 / _freq.QuadPart));   //计算耗时
	printf("total cost time: %.5f\n", costTime);
}
