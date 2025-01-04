#include "opencv2/opencv.hpp"
#include <iostream>
#include "detection.hpp"

using namespace std;
using namespace cv;

const string IMG_PATH = "C:\\Users\\admin\\Desktop\\deep-learning-codes\\OpencvCpp\\assets\\";


void sobel() {
	Mat img = imread(IMG_PATH + "object2.png", IMREAD_GRAYSCALE);
	if (img.empty()) return;

	Mat dx, dy;
	Sobel(img, dx, CV_32FC1, 1, 0);
	Sobel(img, dy, CV_32FC1, 0, 1);

	Mat mag_float, mag;
	magnitude(dx, dy, mag_float);
	mag_float.convertTo(mag, CV_8UC1);

	int threshold = 150;
	Mat edge = mag > threshold;

	imshow("img", img);
	imshow("edge", edge);
}

void canny() {
	Mat img = imread(IMG_PATH + "object2.png", IMREAD_GRAYSCALE);
	if (img.empty()) return;

	Mat dst;
	Canny(img, dst, 100, 200);

	imshow("img", img);
	imshow("dst", dst);
}

void hough() {
	Mat img = imread(IMG_PATH + "object1.png", IMREAD_GRAYSCALE);
	if (img.empty()) return;

	Mat edge;
	Canny(img, edge, 50, 150);

	vector<Vec2f> lines;
	int threshold = 100;
	HoughLines(edge, lines, 1, CV_PI / 180, threshold);

	// draw detected lines in red
	Mat dst;
	cvtColor(edge, dst, COLOR_GRAY2BGR);

	for (size_t i = 0; i < lines.size(); i++) {
		float r = lines[i][0], t = lines[i][1];
		double cos_t = cos(t), sin_t = sin(t);
		double x0 = r * cos_t;
		double y0 = r * sin_t;
		double alpha = 1000;

		Point pt1(cvRound(x0 + alpha * (-sin_t)), cvRound(y0 + alpha * cos_t));
		Point pt2(cvRound(x0 - alpha * (-sin_t)), cvRound(y0 - alpha * cos_t));
		line(dst, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);
	}

	imshow("dst", dst);
 }

void hough_segment() {
	Mat img = imread(IMG_PATH + "object1.png", IMREAD_GRAYSCALE);
	if (img.empty()) return;

	Mat edge;
	Canny(img, edge, 50, 100);

	vector<Vec4i> lines;
	int threshold = 30;
	HoughLinesP(edge, lines, 1, CV_PI / 180, threshold);

	Mat dst;
	cvtColor(edge, dst, COLOR_GRAY2BGR);

	for (Vec4i line_ : lines) {
		line(dst, Point(line_[0], line_[1]), Point(line_[2], line_[3]), 
			Scalar(0, 0, 255), 2, LINE_AA);
	}
	imshow("dst", dst);
}

void hough_circles() {
	Mat img = imread(IMG_PATH + "circle.jpg", IMREAD_GRAYSCALE);
	if (img.empty()) return;

	// blur(img, img, Size(3, 3));

	vector<Vec3f> circles;
	int threshold_canny = 200;
	int threshold_circle = 200;
	HoughCircles(img, circles, HOUGH_GRADIENT, 1, 30, 
		threshold_canny, threshold_circle, 100, 300);

	Mat dst;
	cvtColor(img, dst, COLOR_GRAY2BGR);

	for (Vec3f circle_ : circles) {
		Point center(cvRound(circle_[0]), cvRound(circle_[1]));
		int radius = cvRound(circle_[2]);
		circle(dst, center, radius, Scalar(0, 0, 255), 2, LINE_AA);
	}

	imshow("dst", dst);
}

void hough_circles2() {
	Mat img = imread(IMG_PATH + "circle2.jpg", IMREAD_GRAYSCALE);
	if (img.empty()) return;

	vector<Vec3f> circles;
	int threshold_canny = 200;
	int threshold_circle = 70;
	HoughCircles(img, circles, HOUGH_GRADIENT, 1, 100,
		threshold_canny, threshold_circle, 200, 400);

	Mat dst;
	cvtColor(img, dst, COLOR_GRAY2BGR);

	for (Vec3f circle_ : circles) {
		Point center(cvRound(circle_[0]), cvRound(circle_[1]));
		int radius = cvRound(circle_[2]);
		circle(dst, center, radius, Scalar(0, 0, 255), 2, LINE_AA);
	}

	imshow("dst", dst);
}
