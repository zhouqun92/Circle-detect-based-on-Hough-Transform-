#include "findcircle.h"
void main()
{
	Mat img = imread("2.jpg");
	float center_x, center_y, center_r;
	DWORD timestart = GetTickCount();
	findcircle(img, center_x, center_y, center_r);
	DWORD timeend = GetTickCount();
	cout<< "×ÜÓÃÊ±:" << timeend - timestart << endl;
	circle(img, Point(center_x, center_y), center_r, Scalar(0, 0, 255));
	namedWindow("result",0);
	imshow("result", img); waitKey(0);
	imwrite("result.png", img);
	//system("pause");
}