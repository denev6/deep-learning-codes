#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

const string IMG_PATH = "C:\\Users\\admin\\Desktop\\deep-learning-codes\\OpencvCpp\\src\\";
const Mat LENNA = imread(IMG_PATH + "lenna.bmp");
const Mat LENNA_GRAY = imread(IMG_PATH + "lenna.bmp", IMREAD_GRAYSCALE);

void bgr2gray() {
	Mat src = LENNA;
	Mat dst;
	cvtColor(src, dst, COLOR_BGR2GRAY);

	// or simply:
	// imread(PATH, IMREAD_GRAYSCALE);

	imshow("orginal", src);
	imshow("converted", dst);

	waitKey();
	destroyAllWindows();
}

void brightness_gray() {
	Mat src = imread(IMG_PATH + "img.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) return;

	/* 
	* saturate(x) {
	*	0 if x < 0
	*	255 if x > 0
	*	x else
	* }
	* 
	* x + y == saturate(x + y)
	
	*/

	int change = 100; // random int
	Mat dst = src + change;

	imshow("original", src);
	imshow("brighter", dst);

	waitKey();
	destroyAllWindows();
}

void brightness_manual() {
	Mat src = LENNA_GRAY;
	if (src.empty()) return;

	Mat dst(src.rows, src.cols, src.type());

	int change = 100;

	for (int row = 0; row < src.rows; row++) {
		for (int col = 0; col < src.cols; col++) {
			int v = src.at<uchar>(row, col) + change;
			dst.at<uchar>(row, col) = v > 255 ? 255 : v < 0 ? 0 : v; // saturate

			// or use built-in template:
			// saturate_cast<uchar>(v)
		}
	}

	imshow("original", src);
	imshow("brighter", dst);

	waitKey();
	destroyAllWindows();
}

void brightness_hsv() {
	Mat src = imread(IMG_PATH + "img.jpg", IMREAD_COLOR);
	if (src.empty()) return;

	Mat hsv_image;
	cvtColor(src, hsv_image, cv::COLOR_BGR2HSV);

	vector<Mat> hsv_channels;
	split(hsv_image, hsv_channels);

	int brightness_value = 100;
	hsv_channels[2] = min(hsv_channels[2] + brightness_value, 255);

	merge(hsv_channels, hsv_image);

	Mat dst;
	cvtColor(hsv_image, dst, cv::COLOR_HSV2BGR);

	imshow("original", src);
	imshow("brighter", dst);

	waitKey();
	destroyAllWindows();
}

void contrast() {
	Mat src = imread(IMG_PATH + "img.jpg", IMREAD_COLOR);
	if (src.empty()) return;

	float alpha1 = 1.3f;
	float alpha2 = 0.5f;

	Mat dst1 = src * alpha1;
	Mat dst2 = src * alpha2;

	imshow(format("%.1f", alpha1), dst1);
	imshow(format("%.1f", alpha2), dst2);

	waitKey();
	destroyAllWindows();
}

void contrast2() {
	/*
	* use "*" operater, like "dst = src * 2".
	* But it leads to low quality image 
	* cuz the vectors increase rapidly.
	* 
	* So, use the equation below.
	* dst = src + (src - criteria) * alpha
	* 
	* where 
	* criteria is usually 128(half of 255) or the average of vectors
	* and alpha is a small value; -1 <= alpha.
	* 
	* if -1 <= alpha <= 0; decrease contrast
	* if 0 < alpha; increase contrast
	*/

	Mat src = imread(IMG_PATH + "img.jpg", IMREAD_COLOR);
	if (src.empty()) return;

	float alpha1 = 1.5f;
	float alpha2 = -0.5f;
	int criteria = 128;

	Mat dst1 = src + (src - criteria) * alpha1;
	Mat dst2 = src + (src - criteria) * alpha2;

	imshow(format("%.1f", alpha1), dst1);
	imshow(format("%.1f", alpha2), dst2);

	waitKey();
	destroyAllWindows();
}
