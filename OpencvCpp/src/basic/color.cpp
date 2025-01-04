#include "opencv2/opencv.hpp"
#include <iostream>
#include "basic.hpp"

using namespace std;
using namespace cv;

const string IMG_PATH = "C:\\Users\\admin\\Desktop\\deep-learning-codes\\OpencvCpp\\assets\\";

void bgr2gray() {
	Mat src = imread(IMG_PATH + "img.jpg");
	Mat dst;
	cvtColor(src, dst, COLOR_BGR2GRAY);

	// or simply:
	// imread(PATH, IMREAD_GRAYSCALE);

	imshow("orginal", src);
	imshow("converted", dst);

	waitKey();
	destroyAllWindows();
}

void inverse_color() {
	Mat img = imread(IMG_PATH + "img3.jpg", IMREAD_COLOR);
	if (img.empty()) return;

	Mat dst(img.rows, img.cols, img.type());

	for (int row = 0; row < img.rows; row++) {
		for (int col = 0; col < img.cols; col++) {
			Vec3b& p_img = img.at<Vec3b>(row, col);
			Vec3b& p_dst = dst.at<Vec3b>(row, col);

			p_dst[0] = 255 - p_img[0]; // B
			p_dst[1] = 255 - p_img[1]; // G
			p_dst[2] = 255 - p_img[2]; // R
		}
	}
	imshow("img", img);
	imshow("inversed", dst);
}

void rgb2gray_manual() {
	// https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html

	Mat img = imread(IMG_PATH + "img3.jpg", IMREAD_COLOR);
	if (img.empty()) return;

	Mat dst(img.rows, img.cols, CV_8UC1);
	
	for (int row = 0; row < img.rows; row++) {
		for (int col = 0; col < img.cols; col++) {
			Vec3b pixel = img.at<Vec3b>(row, col);
			uchar blue = pixel[0];
			uchar green = pixel[1];
			uchar red = pixel[2];
			uchar grayscale = static_cast<uchar>(0.299 * red + 0.587 * green + 0.114 * blue);
			dst.at<uchar>(row, col) = grayscale;
		}
	}
	imshow("img", img);
	imshow("grayscale", dst);
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

void mask_hue() {
	Mat img = imread(IMG_PATH + "img2.jpg", IMREAD_COLOR);
	if (img.empty()) return;
	imshow("img", img);

	cvtColor(img, img, COLOR_BGR2HSV);
	int lower_hue = 0;
	int upper_hue = 20;
	Scalar lowerb(lower_hue, 100, 0);
	Scalar upperb(upper_hue, 255, 255);

	Mat mask;
	inRange(img, lowerb, upperb, mask);
	imshow("mask", mask);
}
