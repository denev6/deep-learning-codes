#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

const string IMG_PATH = "C:\\Users\\admin\\Desktop\\deep-learning-codes\\OpencvCpp\\src\\";

void on_level_change(int pos, void* userdata) {
	Mat img = *(Mat*)userdata;

	img.setTo(pos * 16);
	imshow("image", img);
}

void trackbar() {
	Mat img = Mat::zeros(400, 400, CV_8UC1);

	namedWindow("image");
	createTrackbar("level", "image", 0, 16, on_level_change, (void*)&img);
	imshow("image", img);
	waitKey();
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

void overlay() {
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