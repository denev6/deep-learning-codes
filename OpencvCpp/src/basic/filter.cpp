#include "opencv2/opencv.hpp"
#include <iostream>
#include "basic.hpp"

using namespace std;
using namespace cv;

const string IMG_PATH = "C:\\Users\\admin\\Desktop\\deep-learning-codes\\OpencvCpp\\assets\\";

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

void unsharp_mask() {
	Mat src = imread(IMG_PATH + "img.jpg");
	if (src.empty()) return;
	
	Mat blurred;
	GaussianBlur(src, blurred, Size(), 3.0);

	// sharpening image
	float alpha = 1.f;
	Mat dst = (1 + alpha) * src - alpha * blurred;

	imshow("blurred", blurred);
	imshow("sharpen", dst);
}

void noise() {
	Mat src = imread(IMG_PATH + "img.jpg");
	if (src.empty()) return;

	// Gaussian noise model
	Mat noise(src.size(), CV_32SC3);
	int mean = 0;
	int stddev = 30;
	randn(noise, mean, stddev);

	Mat dst;
	add(src, noise, dst, Mat(), CV_8U);

	imshow("original", src);
	imshow("noise", dst);
}

void remove_noise_gaussian() {
	Mat src = imread(IMG_PATH + "img.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) return;
	resize(src, src, Size(300, 300)); // to save computational cost


	Mat noise(src.size(), CV_32SC1);
	/* when src is a COLOR image.
	Mat noise(src.size(), CV_32SC3);
	*/
	randn(noise, 0, 10);
	add(src, noise, src, Mat(), CV_8U);

	// remove noises
	Mat dst;
	int sigma_color = 30;
	int sigma_space = 70;
	bilateralFilter(src, dst, -1, sigma_color, sigma_space);

	imshow("with noise", src);
	imshow("without noise", dst);
}

void remove_noise_median() {
	Mat src = imread(IMG_PATH + "img.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) return;

	// add salt and pepper
	double noise_ratio = 0.1;
	int num_noise = (int)(src.total() * noise_ratio);
	for (int i = 0; i < num_noise; i++) {
		// generate a random spot(x, y)
		int x = rand() % src.cols;
		int y = rand() % src.rows;
		// add noise(0 or 255)
		src.at<uchar>(y, x) = (i % 2) * 255;
	}

	/* noise when "src" is a COLOR image.
	Vec3b pixel = src.at<Vec3b>(y, x);
	for (int c = 0; c < 3; c++) {
		pixel[c] = (i % 2) * 255;
	}
	*/

	// remove noises
	Mat dst;
	int kernel_size = 3;
	medianBlur(src, dst, kernel_size);

	imshow("with noise", src);
	imshow("without noise", dst);
}
