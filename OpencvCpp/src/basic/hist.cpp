#include "opencv2/opencv.hpp"
#include <iostream>
#include "basic.hpp"

using namespace std;
using namespace cv;

const string IMG_PATH = "C:\\Users\\admin\\Desktop\\deep-learning-codes\\OpencvCpp\\assets\\";
const Mat LENNA = imread(IMG_PATH + "lenna.bmp", IMREAD_GRAYSCALE);


Mat calcGrayHist(const Mat& img) {
	CV_Assert(img.type() == CV_8UC1);

	Mat hist;
	int channels[] = { 0 }; // grayscale
	int dims = 1; // # of hists == # of channels
	const int histSize[] = { 256 }; // # of bins for each channels
	float graylevel[] = { 0, 256 }; // range of each vector { min, max }
	const float* ranges[] = { graylevel };

	calcHist(&img, 1, channels, noArray(), hist, dims, histSize, ranges);

	return hist;
}

Mat getGrayHistImage(const Mat& hist) {
	CV_Assert(hist.type() == CV_32FC1);
	CV_Assert(hist.size() == Size(1, 256));

	// max value of hist in "histMax"
	double histMax;
	minMaxLoc(hist, 0, &histMax);

	// init 256 x 100 image to draw a hist on it.
	int histHeight = 100;
	int numBins = 256;
	Mat imgHist(histHeight, numBins, CV_8UC1, Scalar(255));

	for (int bin = 0; bin < numBins; bin++) {
		Point bottom = Point(bin, histHeight);
		Point top = Point(bin, histHeight - cvRound(hist.at<float>(bin, 0) / histMax * histHeight));
		line(imgHist, bottom, top, Scalar(0));
	}

	return imgHist;
}

void drawHist() {
	Mat src = LENNA;
	Mat hist = calcGrayHist(src);
	Mat hist_img = getGrayHistImage(hist);

	imshow("hist", hist_img);
}

void compareBrightnessHist() {
	Mat src = LENNA;
	if (src.empty()) return;

	Mat img_high = src + 30;
	Mat img_low = src - 30;

	Mat hist_high = getGrayHistImage(calcGrayHist(img_high));
	Mat hist_low = getGrayHistImage(calcGrayHist(img_low));

	imshow("high hist", hist_high);
	imshow("low hist", hist_low);
}

void compareContrastHist() {
	Mat src = LENNA;
	if (src.empty()) return;

	Mat img_high = src + (src - 128) * 0.5f;
	Mat img_low = src + (src - 128) * -0.5f;

	Mat hist_high = getGrayHistImage(calcGrayHist(img_high));
	Mat hist_low = getGrayHistImage(calcGrayHist(img_low));

	imshow("high hist", hist_high);
	imshow("low hist", hist_low);
}

void equalizeColorHist() {
	Mat img = imread(IMG_PATH + "img3.jpg", IMREAD_COLOR);
	if (img.empty()) return;
	imshow("img", img);

	cvtColor(img, img, COLOR_BGR2YCrCb);

	vector<Mat> yCrCb_planes;
	split(img, yCrCb_planes);

	equalizeHist(yCrCb_planes[0], yCrCb_planes[0]);

	Mat dst;
	merge(yCrCb_planes, dst);
	cvtColor(dst, dst, COLOR_YCrCb2BGR);

	imshow("dst", dst);
}
