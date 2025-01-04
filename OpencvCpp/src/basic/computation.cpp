#include "opencv2/opencv.hpp"
#include <iostream>
#include "basic.hpp"

using namespace std;
using namespace cv;

const string IMG_PATH = "C:\\Users\\admin\\Desktop\\deep-learning-codes\\OpencvCpp\\assets\\";


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
	Mat src = imread(IMG_PATH + "lenna.bmp", IMREAD_GRAYSCALE);
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

void mask_setTo() {
	Mat src = imread(IMG_PATH + "img.jpg");
	Mat mask = imread(IMG_PATH + "mask.bmp"); // 0 and 255 only

	if (src.empty() || mask.empty()) return;

	// shape(size, channel) of the image and mask must match.
	resize(mask, mask, src.size());

	Scalar green = Scalar(0, 255, 0);
	src.setTo(green, mask); // green if not 0.

	imshow("masked", src);
	imshow("origianl mask", mask);
}

void mask_copyTo() {
	Mat src = imread(IMG_PATH + "img.jpg");
	Mat mask = imread(IMG_PATH + "mask.bmp");
	Mat dst = imread(IMG_PATH + "img2.jpg");

	if (src.empty() || mask.empty() || dst.empty()) return;
	resize(mask, mask, src.size());

	src.copyTo(dst, mask); // copy if not 0.

	imshow("masked", dst);
	imshow("mask", mask);
}

void overlap() {
	Mat img1 = imread(IMG_PATH + "man1.jpg");
	Mat img2 = imread(IMG_PATH + "man2.jpg");

	Mat dst;
	// dst = img1 * 0.5 + img2 * 0.5 + 0;
	addWeighted(img1, 0.5, img2, 0.5, 0, dst);

	imshow("dst", dst);
}

void find_diff() {
	Mat img1 = imread(IMG_PATH + "desk1.jpg");
	Mat img2 = imread(IMG_PATH + "desk2.jpg");

	Mat dst;
	// dst = abs(img1 - img2);
	absdiff(img1, img2, dst);

	imshow("before", img1);
	imshow("after", img2);
	imshow("diff", dst);
}

void mask_grayscale_bitwise() {
	Mat img = imread(IMG_PATH + "img.jpg", IMREAD_GRAYSCALE);

	// mask := { bg: 0, ROI: 255 }
	Mat mask = imread(IMG_PATH + "mask.bmp", IMREAD_GRAYSCALE);
	resize(mask, mask, img.size());

	Mat dst;
	bitwise_and(img, mask, dst);

	imshow("masked", dst);
}
