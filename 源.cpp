#include "findcircle.h"
//The core function:imfindcircles
//Adjustable parameter
//src:input img 
//rmin:Minimum radius
//rmax:Maximum radius
//sensitivity:Larger is easier to detect a circle ,range(0-1)
//edgeThresh:The gradient threshold,range(0-1)
//just detect one of the circle with the highest score,the circle must Brighter than the background.Otherwise you can reverse the image 
//by 255-img if the img is grayscale
void main()
{
	Mat img = imread("2.jpg");
	float center_x, center_y, center_r;
	DWORD timestart = GetTickCount();
	findcircle(img, center_x, center_y, center_r);
	DWORD timeend = GetTickCount();
	cout<< "cost time:" << timeend - timestart << endl;
	circle(img, Point(center_x, center_y), center_r, Scalar(0, 0, 255));
	namedWindow("result",0);
	imshow("result", img); waitKey(0);
	imwrite("result.png", img);
	//system("pause");
}
