#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

const string IMG_PATH = "C:\\Users\\admin\\Desktop\\deep-learning-codes\\OpencvCpp\\src\\";

void embossing() {
	Mat img = imread(IMG_PATH + "object.jpg", IMREAD_GRAYSCALE);
	if (img.empty()) return;

	float kernel[] = {
		-1, -1, 0,
		-1, 0, 1,
		0, 1, 1
	};
	Mat embossFilter(3, 3, CV_32FC1, kernel); // 3 by 3 matrix
	int delta1 = 0;
	int delta2 = 128;

	Mat dst1, dst2;
	filter2D(img, dst1, -1, embossFilter, Point(-1, -1), delta1);
	filter2D(img, dst2, -1, embossFilter, Point(-1, -1), delta2);
	// 5th param "Point(-1, -1)" is the default value 
	// that means the anchor is at the center.

	imshow("img", img);
	imshow("delta 0", dst1);
	imshow("delta 128", dst2);
}

void mean_blur() {
	Mat img = imread(IMG_PATH + "img.jpg");
	if (img.empty()) return;

	Mat dst;
	int kernelSize = 5;
	// sum(adjacent vectors) / (number of vectors)
	blur(img, dst, Size(kernelSize, kernelSize));

	imshow("img", img);
	imshow("blurred", dst);
}

void gaussian_blur() {
	Mat img = imread(IMG_PATH + "img.jpg");
	if (img.empty()) return;

	Mat dst;
	int sigma = 3;
	GaussianBlur(img, dst, Size(), (double)sigma);
	// 3rd param "Size()" will automatically choose the proper size 
	// of Gaussian kernel using "sigma".

	imshow("img", img);
	imshow("blurred", dst);
}
