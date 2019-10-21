#pragma once
//#include "HalconCpp.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "iostream"
#include <vector>
#include <algorithm>
#include <opencv2\opencv.hpp>  
#include <Windows.h>
#include <omp.h>
#include <queue>
#include <math.h>
//#include<thread>
using namespace cv;
using namespace std;
//using namespace HalconCpp;
int x_offset[8] = { 1,1,1,0,0,-1,-1,-1 };
int y_offset[8] = { 1,-1,0,-1,1,-1,1,0 };
#define NEGHTBOR 8
//Mat HObject2Mat(HObject Hobj);
Mat findpeeks(Mat I, Mat J);
bool isexistNBzero(Mat im, Point p);
Mat regionalmaximaImg(Mat im);
float regionalmaxima(Mat im, Point p);
Mat accumarray(Mat subs, Mat val, int M, int N);
Mat ImgIndex2(Mat img, Mat idx);
Mat IdenxRows(Mat img, Mat rows_to_keep);
Mat cvRoundImage(Mat img);
Mat ImgIndex(Mat img, Mat idx);
void chaccum(Mat img, double rmin, double rmax, double edgeThresh, Mat &accumMatrix, Mat &gradientImg);
void find(Mat gradientImg, float t, vector<int> &Ex, vector<int> &Ey, vector<int> &idxE);
void getEdgePixels(Mat gradientImg, float edgeThresh, vector<int>  &Ex, vector<int>  &Ey, vector<int>&idxE);
void GetgradientImage(Mat img, Mat &Gx, Mat &Gy, Mat &gradientImg);
void chcenters(Mat accumMatrix, double  accumThresh, vector<Point2f> &center, vector<float> &metric);
void imfindcircles(Mat src, double rmin, double rmax, double sensitivity, double edgeThresh, vector<Point2f>&select_center, vector<float>&r_estimated, vector<float>& metric_select);
void exp_complex(Mat phi, Mat &Opca);
void imhmax(Mat& Hd, float suppThreshold);
Mat peak_local_max(Mat image);
void WeightCentroid(Mat bw, Mat Hd, vector<Point2f> &center, vector<float> &metric);
void chradiiphcode(vector<Point2f>centers, Mat accumMatrix, float rmin, float rmax, vector<float> &r_estimated);
void keep_(Mat xc, Mat yc, Mat w, int M, int N, Mat& xc_keep, Mat& yc_keep, Mat& w_keep, Mat accumMatrix_temp);
void findcircle(Mat src, float &center_x, float &center_y, float &center_r);
Mat  Repeat(Mat m, int rows, int cols); 


double round(double r)
{
	return (r>0.0)?floor(r+0.5):ceil(r-0.5);
}




void findcircle(Mat src, float &center_x, float& center_y, float& center_r)
{
	double edgeThresh = 0.01;
	double sensitivity = 0.999;
	vector<Point2f>select_center; vector<float> r_estimated;
	vector<float> metric_select;
	//Mat src = HObject2Mat(Himg);//namedWindow("SRC",0);imshow("SRC",src);waitKey(0);WriteImage(Himg, "bmp", 0, "G:/d/4");
	if (src.channels() == 3)
	{
		cvtColor(src, src, COLOR_BGR2GRAY);
	}
	Mat src_float;
	//src = 255 - src;
	src.convertTo(src_float, CV_32FC1);
	src_float = src_float / 255.0;
	imfindcircles(src_float, 450, 650, sensitivity, edgeThresh, select_center, r_estimated, metric_select);
	int max_id; float max_metric = 0;
	if (metric_select.size() > 0)
	{
		for (int i = 0; i < metric_select.size(); i++)
		{
			if (metric_select[i] > max_metric)
			{
				max_metric = metric_select[i];
				max_id = i;
			}
		}	
	center_x = select_center[max_id].x;
	center_y = select_center[max_id].y;
	center_r = r_estimated[max_id];
	}
}

void imfindcircles(Mat src, double rmin, double rmax, double sensitivity, double edgeThresh, vector<Point2f>&select_center, vector<float>&r_estimated, vector<float>&metric_select)//实现找圆函数
{
	Mat accumMatrix, gradientImg;
	vector<Point2f> center;
	vector<float> metric;
	chaccum(src, rmin, rmax, edgeThresh, accumMatrix, gradientImg);
	double accumThresh = 1 - sensitivity;
	//DWORD timestart = GetTickCount();
	chcenters(accumMatrix, accumThresh, center, metric);
	//DWORD timeend = GetTickCount();
	//cout << "求圆心耗时:" << timeend - timestart << endl;
	//	r_estimated = chradiiphcode(centers, accumMatrix, rmin, rmax);
	for (int i = 0; i < center.size(); i++)
	{
		if (metric[i] > accumThresh)
		{
			select_center.push_back(center[i]);
			metric_select.push_back(metric[i]);
		}
	}
	chradiiphcode(select_center, accumMatrix, rmin, rmax, r_estimated);

}
void chcenters(Mat accumMatrix, double  accumThresh, vector<Point2f> &center, vector<float> &metric)
{
	int medFiltSize = 5;
	vector <Mat> channels;
	split(accumMatrix, channels);
	Mat accumMatrix_abs;
	sqrt((channels.at(0)).mul(channels.at(0)) + (channels.at(1)).mul(channels.at(1)), accumMatrix_abs);
	Mat Hd;
	medianBlur(accumMatrix_abs, Hd, medFiltSize);
	float suppThreshold = 0.01;
	imhmax(Hd, suppThreshold);
	//Mat bw = findpeeks(Hd+0.00001, Hd +0.00001- 0.00001);
	Mat bw = peak_local_max(Hd);
	//cout << Hd << endl;
	WeightCentroid(bw, accumMatrix_abs, center, metric);
}
void GetgradientImage(Mat img, Mat &Gx, Mat &Gy, Mat &gradientImg)
{
	float mask_y[3][3] = { { 1,2,1 },{ 0,0,0 },{ -1,-2,-1 } };
	Mat y_mask = Mat(3, 3, CV_32F, mask_y);// / 8;
	float mask_x[3][3] = { { 1,0,-1 },{ 2,0,-2 },{ 1,0,-1 } };
	Mat x_mask = Mat(3, 3, CV_32F, mask_x); // 转置


											// 计算x方向和y方向上的滤波
	filter2D(img, Gx, CV_32F, x_mask);
	filter2D(img, Gy, CV_32F, y_mask);
	Mat Gx_abs, Gy_abs;
	Gx_abs = abs(Gx);
	Gy_abs = abs(Gy);
	sqrt(Gx_abs.mul(Gx_abs) + Gy_abs.mul(Gy_abs), gradientImg);
}

void getEdgePixels(Mat gradientImg, float edgeThresh, vector<int>  &Ex, vector<int>  &Ey, vector<int>&idxE)
{
	double Gmin, Gmax;
	minMaxLoc(gradientImg, &Gmin, &Gmax);//正确 
										 //cout << Gmax << endl;
	float t = Gmax * edgeThresh;
	find(gradientImg, t, Ex, Ey, idxE);//正确索引比matlab的少一
}
//实现matlab find函数
void find(Mat gradientImg, float t, vector<int> &Ex, vector<int> &Ey, vector<int> &idxE)
{
	int rows = gradientImg.rows;
	int cols = gradientImg.cols;
	// #pragma omp parallel for
	for (int i = 0; i < rows; i++)
	{
		float* gradientImgdata = gradientImg.ptr<float>(i);
		for (int j = 0; j < cols; j++)
		{
			if (gradientImgdata[j]>t)
			{
				Ey.push_back(i);
				Ex.push_back(j);
				idxE.push_back(j*rows + i);
				//cout << j*rows + i << endl;
			}

		}
	}
}

void chaccum(Mat img, double rmin, double rmax, double edgeThresh, Mat &accumMatrix, Mat &gradientImg)
{
	vector<int> Ex;
	//Mat A;
	vector<int> Ey;
	//Mat Ex, Ey;
	vector<int> idxE;
	Mat Gx, Gy;
	//DWORD timestart = GetTickCount();
	GetgradientImage(img, Gx, Gy, gradientImg);//正确耗时严重
	getEdgePixels(gradientImg, edgeThresh, Ex, Ey, idxE);//正确耗时严重
	/*DWORD timeend = GetTickCount();
	cout << "求梯度耗时:" << timeend - timestart << endl;*/
	int Range[2] = { rmin,rmax };
	int size = floor((Range[1] - Range[0]) / 0.5) + 1;
	Mat radiusRange, RR, lnR, phi, Opca, w0;
	radiusRange.create(1, size, CV_32FC1);
	for (int i = 0; i < size; i++)
		radiusRange.at<float>(0, i) = Range[0] + i*0.5;//正确
													   //cout << radiusRange;
	radiusRange.copyTo(RR);
	log(radiusRange, lnR);  //正确
							//cout << lnR << endl;
	phi.create(1, size, CV_32FC1);
	Opca.create(1, size, CV_32FC2);
	w0.create(1, size, CV_32FC2);
	phi = ((lnR - lnR.at<float>(0, 0)) / (lnR.at<float>(0, size - 1) - lnR.at<float>(0, 0)) * 2 * CV_PI) - CV_PI;//正确
																												 //cout << phi << endl;
	exp_complex(phi, Opca);//正确
	vector <Mat>channels, w0_channels;
	Mat img_channel1, img_channel2;
	split(Opca, channels);
	img_channel1 = channels.at(0);
	img_channel2 = channels.at(1);
	Mat temp1 = 1 / (2 * CV_PI*radiusRange);
	w0_channels.push_back(img_channel1.mul(temp1));
	w0_channels.push_back(img_channel2.mul(temp1));
	merge(w0_channels, w0);//正确
						   //cout << w0 << endl;
	double maxNumElemNHoodMat = 1e7;
	int xcStep = floor(maxNumElemNHoodMat / size / 3);//////
	int lenE = Ex.size();
	int M = img.rows;
	int N = img.cols;
	accumMatrix = Mat::zeros(M, N, CV_32FC2);
	Mat Ex_mat = Mat(Ex);
	Mat Ey_mat = Mat(Ey);
	Mat idxE_mat = Mat(idxE);
	//cout << idxE_mat << endl;
	//timestart = GetTickCount();
    #pragma omp parallel for
	for (int i = 0; i < lenE; i = i + xcStep)
	{
		
		//cout << min(i + xcStep - 1, lenE - 1) << endl;
		Mat Ex_chunk = Ex_mat(cv::Range(i, min(i + xcStep - 1, lenE)), cv::Range(0, 1));///??
		Mat Ey_chunk = Ey_mat(cv::Range(i, min(i + xcStep - 1, lenE)), cv::Range(0, 1));
		Mat idxE_chunk = idxE_mat(cv::Range(i, min(i + xcStep - 1, lenE)), cv::Range(0, 1));//正确
		Mat gradientImg_idxE = ImgIndex(gradientImg, idxE_chunk);//正确
																 //cout << Ex_chunk << endl;
		            											 //gradientImg_idxE = gradientImg_idxE.t();
		Mat Gx_idxE = ImgIndex(Gx, idxE_chunk);
		//cout << Gy << endl;
		//Gx_idxE = Gx_idxE.t();
		Mat Gy_idxE = ImgIndex(Gy, idxE_chunk);
		//Gy_idxE = Gy_idxE.t();
		Mat Gx_mul_gradientImg = Gx_idxE.mul(1 / gradientImg_idxE);
		//cout << gradientImg_idxE << endl;
		Mat Gy_mul_gradientImg = Gy_idxE.mul(1 / gradientImg_idxE);
		Mat repeat_img1 = repeat(RR, Gx_mul_gradientImg.rows, 1);
		Mat Ex_chunk_repeat, times1, times2, Ey_chunk_repeat;
        #pragma omp parallel for
		for (int i = 0; i < 2; i++)
		{
			if (i == 1)
			{
				//Mat repeat_img2 = repeat(Gx_mul_gradientImg, 1, RR.cols);//60ms
				Mat repeat_img2 = repeat(Gx_mul_gradientImg, 1, RR.cols);
				times1 = repeat_img1.mul(-repeat_img2);
				Ex_chunk_repeat = repeat(Ex_chunk, 1, RR.cols);//42ms
			}
			else
			{
				Mat repeat_img3 = repeat(Gy_mul_gradientImg, 1, RR.cols);//60ms
				times2 = repeat_img1.mul(-repeat_img3);
				Ey_chunk_repeat = repeat(Ey_chunk, 1, RR.cols);//44ms
			}
		}
		//cout << times1 << endl;
		Ex_chunk_repeat.convertTo(Ex_chunk_repeat, CV_32FC1);
		Ey_chunk_repeat.convertTo(Ey_chunk_repeat, CV_32FC1);
		Mat xc = Ex_chunk_repeat + times1;
		Mat yc = Ey_chunk_repeat + times2;
		//耗时太多 需要优化
		//xc = cvRoundImage(xc);
		//yc = cvRoundImage(yc);//正确
							  //
							  //cout << yc << endl;
		Mat w = repeat(w0, xc.rows, 1);

		Mat  xc_keep, yc_keep, w_keep;
		Mat accumMatrix_temp = Mat::zeros(Size(N, M), CV_32FC2);
		keep_(xc, yc, w, M, N, xc_keep, yc_keep, w_keep, accumMatrix_temp);
		//Mat inside =(( xc >= 0 )& (xc < N)&(yc >= 0) & (yc < M - 1));//正确
		//inside.convertTo(inside, CV_32FC1);
		////cout << ((xc >= 0)& (xc < N)&(yc >= 0) ) << endl;
		//Mat inside_sum_col;
		//reduce(inside, inside_sum_col, 2, CV_REDUCE_SUM);
		//Mat rows_to_keep = inside_sum_col > 0;
		//xc = IdenxRows(xc, rows_to_keep);
		//yc = IdenxRows(yc, rows_to_keep);
		//w= IdenxRows(w, rows_to_keep);
		////cout << inside << endl;
		//inside= IdenxRows(inside, rows_to_keep);	
		//////////???????ImgIndex2有问题  耗时太多
		//xc = ImgIndex2(xc, inside);//100多ms
		//yc = ImgIndex2(yc, inside);//100多ms
		////cout << xc << endl;


		//Mat xy_cat;
		//hconcat(yc_keep, xc_keep, xy_cat);
		//cout << ImgIndex2(w, inside) << endl;
		//Mat accumMatrix_temp = accumarray(xy_cat, ImgIndex2(w, inside), M, N);
		//Mat accumMatrix_temp = accumarray(xy_cat, w_keep, M, N);
		//accumMatrix += accumarray(xy_cat, ImgIndex2(w, inside), M, N);
		accumMatrix += accumMatrix_temp;
		//cout << accumMatrix << endl;		
	}
	/*timeend = GetTickCount();
	cout << "迭代循环:" << timeend - timestart << endl;*/
}

void exp_complex(Mat phi, Mat &Opca)
{
	//#pragma omp parallel for
	for (int i = 0; i < phi.cols; i++)
	{
		Opca.at<Vec2f>(0, i)[0] = cos(phi.at<float>(0, i));
		Opca.at<Vec2f>(0, i)[1] = sin(phi.at<float>(0, i));
	}
}
Mat ImgIndex(Mat img, Mat idx)
{
	int img_rows = img.rows;
	int img_cols = img.cols;
	int idx_rows = idx.rows;
	int idx_cols = idx.cols;
	Mat dist;
	// #pragma omp parallel for
	for (int i = 0; i < idx_rows; i++)
	{
		int x = (idx.at<int>(i, 0)) / img_rows;
		int y = (idx.at<int>(i, 0)) % img_rows;
		//cout << img.at<float>(y, x) << endl;
		dist.push_back(img.at<float>(y, x));
	}
	return dist;
}
Mat cvRoundImage(Mat img)
{
	Mat output;
	img.copyTo(output);
	int img_rows = img.rows;
	int img_cols = img.cols;
#pragma omp parallel for
	for (int i = 0; i < img_rows; i++)
	{
		float* outData = output.ptr<float>(i);
		float* imgData = img.ptr<float>(i);
		for (int j = 0; j < img_cols; j++)
		{
			outData[j] = round(imgData[j]);
			//output.at<float>(i, j) = round(img.at<float>(i, j));
		}
	}
	return output;
}
Mat IdenxRows(Mat img, Mat rows_to_keep)
{
	Mat dst;
	for (int i = 0; i < img.rows; i++)
	{
		if (rows_to_keep.at<uchar>(i, 0) != 0)
		{
			Mat temp = img.row(i);
			dst.push_back(temp);
		}

	}
	return dst;
}

Mat ImgIndex2(Mat img, Mat idx)
{
	int img_rows = img.rows;
	int img_cols = img.cols;
	int idx_rows = idx.rows;
	int idx_cols = idx.cols;
	Mat dist; Mat dist_1, dist_2;
	//#pragma omp parallel for
	for (int i = 0; i < img_rows; i++)
	{
		//#pragma omp parallel for
		int* idxData = idx.ptr<int>(i);
		float* imgData = img.ptr<float>(i);
		for (int j = 0; j < img_cols; j++)
		{
			if (idxData[j] != 0)
			{
				if (img.channels() == 1)
					dist.push_back(imgData[j]);
				else
				{

					dist_1.push_back(imgData[j * 2]);
					dist_2.push_back(imgData[j * 2 + 1]);
				}
			}
		}
	}
	if (img.channels() == 2)
	{
		vector<Mat> chaneels;
		chaneels.push_back(dist_1);
		chaneels.push_back(dist_2);
		merge(chaneels, dist);
	}
	return dist;// .t();
}

Mat accumarray(Mat subs, Mat val, int M, int N)
{
	//cout << subs << endl;
	Mat out = Mat::zeros(Size(N, M), CV_32FC2);
#pragma omp parallel for
	for (int i = 0; i < subs.rows; i++)
	{
		//subs.convertTo(subs,CV_8UC1);
		float* valData = val.ptr<float>(i);
		out.at<Vec2f>(int(subs.at<float>(i, 0)), int(subs.at<float>(i, 1)))[0] += val.at<Vec2f>(i, 0)[0];
		out.at<Vec2f>(int(subs.at<float>(i, 0)), int(subs.at<float>(i, 1)))[1] += val.at<Vec2f>(i, 0)[1];
		//out.at<Vec2f>(int(subs.at<float>(i, 0)), int(subs.at<float>(i, 1)))[0] += valData[0];
		//out.at<Vec2f>(int(subs.at<float>(i, 0)), int(subs.at<float>(i, 1)))[1] += valData[1];

	}
	/*vector <Mat>channels;
	Mat img_channel1;
	split(out, channels);
	img_channel1 = channels.at(0);
	cout << img_channel1 << endl;*/
	return out;
}




/*寻找最大值邻近点*/
float regionalmaxima(Mat im, Point p)
{
	//cout << im << endl;
	float maxima = im.at<float>(p.y, p.x);
	for (int i = 0; i<NEGHTBOR; i++)
	{
		int x = p.x + x_offset[i];
		int y = p.y + y_offset[i];
		if (x>im.cols - 1) x = im.cols - 1;
		if (x<0)         x = 0;
		if (y>im.rows - 1) y = im.rows - 1;
		if (y<0)         y = 0;
		if (im.at<float>(y, x)>maxima)
			maxima = im.at<float>(y, x);
	}
	return maxima;
}

/*返回最大值滤波图像*/
Mat regionalmaximaImg(Mat im)
{
	Mat result = im.clone();
#pragma omp parallel for
	for (int i = 0; i<im.rows; i++)
		for (int j = 0; j<im.cols; j++)
			result.at<float>(i, j) = regionalmaxima(im, Point(j, i));
	return result;

}
/*邻近点是否存零点*/
bool isexistNBzero(Mat im, Point p)
{
	for (int i = 0; i<NEGHTBOR; i++)
	{
		int x = p.x + x_offset[i];
		int y = p.y + y_offset[i];
		if (x>im.cols - 1)
			x = im.cols - 1;
		if (x<0)
			x = 0;
		if (y>im.rows - 1)
			y = im.rows - 1;
		if (y<0)
			y = 0;
		if (im.at<float>(y, x) == 0)
			return true;
	}
	return false;
}
Mat findpeeks(Mat I, Mat J)
{
	queue<Point> que;
	/*最大值滤波*/
	J = regionalmaximaImg(J);
	/*对满足条件的点入队列，初始化队列*/
	for (int i = 0; i<I.rows; i++)
		for (int j = 0; j<I.cols; j++)
			if (J.at<float>(i, j) && isexistNBzero(J, Point(j, i)))
				que.push(Point(j, i));
	/*循环队列，直至队列为空*/
	while (!que.empty())
	{
		Point p = que.front();
		que.pop();

		for (int i = 0; i<NEGHTBOR; i++)
		{
			/*get the neighbor point*/
			Point q;
			q.x = p.x + x_offset[i];
			q.y = p.y + y_offset[i];
			if (q.x>I.cols - 1) q.x = I.cols - 1;
			if (q.x<0)       q.x = 0;
			if (q.y>I.rows - 1) q.y = I.rows - 1;
			if (q.y<0)       q.y = 0;

			if (J.at<float>(p.y, p.x)<J.at<float>(q.y, q.x) &&
				I.at<float>(q.y, q.x) != I.at<float>(q.y, q.x))
			{
				J.at<float>(p.y, p.x) = min(J.at<float>(p.y, p.x),
					I.at<float>(q.y, q.x));
				que.push(p);
			}
		}
	}
	return (I - J)>0;
}

//Mat peeks = findpeeks(dis, dis - 1);

void imhmax(Mat& Hd, float suppThreshold)
{
	double minval, maxval;
	Point min_p, max_p;
	minMaxLoc(Hd, &minval, &maxval, &min_p, &max_p);
	float thresh_value = maxval - suppThreshold;
	for (int i = 0; i < Hd.rows; i++)
		for (int j = 0; j < Hd.cols; j++)
			if (Hd.at<float>(i, j)>thresh_value)
				Hd.at<float>(i, j) = thresh_value;

}

Mat peak_local_max(Mat image)
{
	int exclude_border = 1;
	Mat image_max = regionalmaximaImg(image);
	Mat mask = Mat::zeros(image_max.size(), CV_8UC1); 
	mask = image_max == image;
	//for (int i = 1; i < mask.rows - 1; i++)
	//{
	//	float *image_maxdata = image_max.ptr<float>(i);
	//	float *imagedata = image.ptr<float>(i);
	//	uchar *maskdata = mask.ptr<uchar>(i);
	//	//#pragma omp parallel for
	//	for (int j = 1; j < mask.cols - 1; j++)
	//	{
	//		if (image_maxdata[j] == imagedata[j])
	//		{
	//			maskdata[j] = 255;
	//		}
	//	}
	//}
	float thresholds;
	double min, max;
	Point minid, maxid;
	minMaxLoc(image, &min, &max, &minid, &maxid);
	thresholds = min;
	mask = (image > thresholds)&mask;
	//cout << mask << endl;
	return mask;
}

void WeightCentroid(Mat bw, Mat Hd, vector<Point2f> &center, vector<float> &metric)
{
	Mat Imglabels, Imgstats, Imgcentriods;
	int Imglabelnum = connectedComponentsWithStats(bw, Imglabels, Imgstats, Imgcentriods);
	//cout << Imglabels<< endl;
	float *M = new float[Imglabelnum];
	float *Wx = new float[Imglabelnum];
	float *Wy = new float[Imglabelnum];
	for (int i = 0; i < Imglabelnum; i++)
	{
		M[i] = 0;
		Wx[i] = 0;
		Wy[i] = 0;
	}
	for (int i = 0; i < Imglabels.rows; i++)
	{
		int *Imglabelsdata = Imglabels.ptr<int>(i);
		float *Hddata = Hd.ptr<float>(i);
		for (int j = 0; j < Imglabels.cols; j++)
		{
			if (Imglabelsdata[j]>0)
			{
				int label_id = Imglabelsdata[j]; //cout << Imglabels.at<int>(i, j) << endl;
				M[label_id] = M[label_id] + Hddata[j];
				Wx[label_id] = Wx[label_id] + j * Hddata[j];
				Wy[label_id] = Wy[label_id] + i * Hddata[j];
			}

		}
	}
	//cout << M[1] << endl; cout << Wx[1] << endl; cout << Wy[1] << endl;
	for (int i = 1; i < Imglabelnum; i++)
	{
		Wx[i] = Wx[i] / M[i];
		Wy[i] = Wy[i] / M[i];
		/*Point2f A;
		A.x = Wx;
		A.y = Wy;*/
		if (0 < cvRound(Wy[i])&cvRound(Wy[i]) < Hd.rows & 0 < cvRound(Wx[i])&cvRound(Wx[i]) < Hd.cols)
		{
			center.push_back(Point2f(Wx[i], Wy[i]));
			metric.push_back(Hd.at<float>(cvRound(Wy[i]), cvRound(Wx[i])));
		}
	}
	//circle(dst, A, 5, Scalar(255, 255, 0), -1, 8, 0);
}

void chradiiphcode(vector<Point2f>centers, Mat accumMatrix, float rmin, float rmax, vector<float> &r_estimated)
{
	for (int i = 0; i < centers.size(); i++)
	{
		float cenPhase = atan2(accumMatrix.at<Vec2f>(cvRound(centers[i].y), cvRound(centers[i].x))[1], accumMatrix.at<Vec2f>(cvRound(centers[i].y), cvRound(centers[i].x))[0]);
		float lnR1 = log(rmin);
		float lnR2 = log(rmax);
		float r_ = exp(((cenPhase + CV_PI) / (2 * CV_PI)*(lnR2 - lnR1)) + lnR1);
		r_estimated.push_back(r_);
	}

}

void keep_(Mat xc, Mat yc, Mat w, int M, int N, Mat& xc_keep, Mat& yc_keep, Mat& w_keep, Mat accumMatrix_temp)
{
	Mat w1_keep, w2_keep;
#pragma omp parallel for
	for (int i = 0; i < xc.rows; i++)
	{
		float *xcdata = xc.ptr<float>(i);
		float *ycdata = yc.ptr<float>(i);
		float *wdata = w.ptr<float>(i);
		for (int j = 0; j < yc.cols; j++)
		{
			if ((xcdata[j] >= 0)& (xcdata[j] < N)&(ycdata[j] >= 0) & (ycdata[j] < M - 1))
			{
				/*xc_keep.push_back(xcdata[j]);
				yc_keep.push_back(ycdata[j]);
				w1_keep.push_back(wdata[j*2]);
				w2_keep.push_back(wdata[j*2+1]);*/

				accumMatrix_temp.at<Vec2f>(cvRound(ycdata[j]), cvRound(xcdata[j]))[0] += wdata[j * 2];
				accumMatrix_temp.at<Vec2f>(cvRound(ycdata[j]), cvRound(xcdata[j]))[1] += wdata[j * 2 + 1];

			}
		}
	}
	/*vector<Mat> chaneels;
	chaneels.push_back(w1_keep);
	chaneels.push_back(w2_keep);
	merge(chaneels, w_keep);*/
}


Mat  Repeat(Mat m, int rows, int cols) {

	Mat obj = Mat::zeros(m.rows * rows, m.cols * cols, m.type());
   #pragma omp parallel for
	for (int i = 0; i < rows; i++) {

		for (int j = 0; j < cols; j++) {

			Mat r(obj, Rect(Point(j * m.cols, i * m.rows), Point(j * m.cols + m.cols, i * m.rows + m.rows)));
			m.copyTo(r);
		}
	}
	return  obj;
}

//cv::Mat HObject2Mat(HObject Hobj)
//{
//	HTuple htCh;
//	HString cType;
//	cv::Mat Image;
//	ConvertImageType(Hobj, &Hobj, "byte");
//	CountChannels(Hobj, &htCh);
//	Hlong wid = 0;
//	Hlong hgt = 0;
//	if (htCh[0].I() == 1)
//	{
//		HImage hImg(Hobj);
//		void *ptr = hImg.GetImagePointer1(&cType, &wid, &hgt);//GetImagePointer1(Hobj, &ptr, &cType, &wid, &hgt);  
//		int W = wid;
//		int H = hgt;
//		Image.create(H, W, CV_8UC1);
//		unsigned char *pdata = static_cast<unsigned char *>(ptr);
//		memcpy(Image.data, pdata, W*H);
//	}
//	else if (htCh[0].I() == 3)
//	{
//		void *Rptr;
//		void *Gptr;
//		void *Bptr;
//		HImage hImg(Hobj);
//		hImg.GetImagePointer3(&Rptr, &Gptr, &Bptr, &cType, &wid, &hgt);
//		int W = wid;
//		int H = hgt;
//		Image.create(H, W, CV_8UC3);
//		vector<cv::Mat> VecM(3);
//		VecM[0].create(H, W, CV_8UC1);
//		VecM[1].create(H, W, CV_8UC1);
//		VecM[2].create(H, W, CV_8UC1);
//		unsigned char *R = (unsigned char *)Rptr;
//		unsigned char *G = (unsigned char *)Gptr;
//		unsigned char *B = (unsigned char *)Bptr;
//		memcpy(VecM[2].data, R, W*H);
//		memcpy(VecM[1].data, G, W*H);
//		memcpy(VecM[0].data, B, W*H);
//		cv::merge(VecM, Image);
//	}
//	return Image;
//}
