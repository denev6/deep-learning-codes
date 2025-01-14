#include "opencv2/opencv.hpp"
#include <iostream>
#include "transform.hpp"

using namespace std;
using namespace cv;

const string IMG_PATH = "C:\\Users\\admin\\Desktop\\deep-learning-codes\\OpencvCpp\\assets\\";

void shift() {
	Mat img = imread(IMG_PATH + "img.jpg");
	if (img.empty()) return;
	double d_x = 100;
	double d_y = 150;
	Mat affine_matrix = Mat_<double>(
		{ 2, 3 }, { 1, 0, d_x, 0, 1, d_y }
	);
	Mat dst;
	warpAffine(img, dst, affine_matrix, Size());

	imshow("dst", dst);
}

void shear_x() {
	Mat img = imread(IMG_PATH + "img.jpg");
	if (img.empty()) return;

	double m_x = 0.5;
	Mat affine_matrix = Mat_<double>(
		{ 2, 3 }, { 1, m_x, 0, 0, 1, 0 }
	);
	int x = img.cols;
	int y = img.rows;
	Size dst_size = Size(cvRound(x + y * m_x), y);
	Mat dst;
	warpAffine(img, dst, affine_matrix, dst_size);

	imshow("dst", dst);
}

void shear_y() {
	Mat img = imread(IMG_PATH + "img.jpg");
	if (img.empty()) return;

	double m_y = 0.5;
	Mat affine_matrix = Mat_<double>(
		{ 2, 3 }, { 1, 0, 0, m_y, 1, 0 }
	);
	int x = img.cols;
	int y = img.rows;
	Size dst_size = Size(x, cvRound(y + x * m_y));
	Mat dst;
	warpAffine(img, dst, affine_matrix, dst_size);

	imshow("dst", dst);
}

void scale() {
	Mat img = imread(IMG_PATH + "img.jpg");
	if (img.empty()) return;
	double s_x = 0.7;
	double s_y = 1.3;

	Mat affine_matrix = Mat_<double>(
		{ 2, 3 }, { s_x, 0, 0, 0, s_y, 0 }
	);
	Size dst_size = Size(cvRound(img.cols * s_x), cvRound(img.rows * s_y));
	Mat dst;
	// default Interpolation: `INTER_LINEAR`
	warpAffine(img, dst, affine_matrix, dst_size, INTER_LINEAR);

	imshow("img", img);
	imshow("dst", dst);
}

void rotate_top() {
	Mat img = imread(IMG_PATH + "img.jpg");
	if (img.empty()) return;

	double angle = 15;
	double radian = angle * CV_PI / 180;

	// clockwise rotation matrix when the anchor is (0, 0)
	Mat rotation_matrix = Mat_<double>(
		{2, 3}, {cos(radian), -sin(radian), 0, sin(radian), cos(radian), 0}
	);

	Mat dst;
	warpAffine(img, dst, rotation_matrix, Size());

	imshow("rotated", dst);
}

void rotate_center() {
	Mat img = imread(IMG_PATH + "img.jpg");
	if (img.empty()) return;

	Point2f center(img.cols / 2.f, img.rows / 2.f);
	double angle = 15;

	// easier way to get rotation matrix
	Mat rotation_matrix = getRotationMatrix2D(center, angle, 1);

	Mat dst;
	warpAffine(img, dst, rotation_matrix, Size());

	imshow("rotated", dst);
}

void flip_vertical() {
	Mat img = imread(IMG_PATH + "img.jpg");
	if (img.empty()) return;
	double h = img.rows - 1;

	Mat affine_matrix = Mat_<double>(
		{ 2, 3 }, { 1, 0, 0, 0, -1, h }
	);

	Mat dst;
	warpAffine(img, dst, affine_matrix, Size());
	// filp(img, dst, 0);

	imshow("fliped", dst);
}

void flip_horizontal() {
	Mat img = imread(IMG_PATH + "img.jpg");
	if (img.empty()) return;
	double w = img.cols - 1;

	Mat affine_matrix = Mat_<double>(
		{ 2, 3 }, { -1, 0, w, 0, 1, 0 }
	);

	Mat dst;
	warpAffine(img, dst, affine_matrix, Size());
	// filp(img, dst, 1);

	imshow("fliped", dst);
}

void perspective_transform() {
	Mat img = imread(IMG_PATH + "object3.jpg");
	if (img.empty()) return;

	// when four coordinates are given
	Point2f objectPoint[4] = {
		Point2f(10, 141),
		Point2f(212, 29),
		Point2f(486, 273),
		Point2f(268, 477)
	};

	int dst_w = cvRound(img.cols / 2);
	int dst_h = cvRound(img.rows / 1.5);
	Point2f dstPoint[4] = {
		Point2f(0, 0),
		Point2f(dst_w - 1, 0),
		Point2f(dst_w - 1, dst_h - 1),
		Point2f(0, dst_h - 1)
	};
	Mat dst;
	
	Mat transform_matrix = getPerspectiveTransform(objectPoint, dstPoint);
	warpPerspective(img, dst, transform_matrix, Size(dst_w, dst_h));

	imshow("img", img);
	imshow("dst", dst);
}
