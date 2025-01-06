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

void scharr() {
	Mat img = imread(IMG_PATH + "object2.png", IMREAD_GRAYSCALE);
	if (img.empty()) return;

	Mat dx, dy;
	Scharr(img, dx, CV_32FC1, 1, 0);
	Scharr(img, dy, CV_32FC1, 0, 1);

	Mat mag_float, mag;
	magnitude(dx, dy, mag_float);
	mag_float.convertTo(mag, CV_8UC1);

	int threshold = 250;
	Mat edge = mag > threshold;

	imshow("img", img);
	imshow("edge", edge);
}

Mat apply_NMS(const Mat& magnitude, const Mat& angle) {
	Mat nms = Mat::zeros(magnitude.size(), CV_8UC1);

	for (int y = 1; y < magnitude.rows - 1; ++y) {
		for (int x = 1; x < magnitude.cols - 1; ++x) {
			float angle_deg = angle.at<float>(y, x);
			angle_deg = fmod(angle_deg, 180.0f);

			float mag = magnitude.at<float>(y, x);

			float mag1 = 0.0, mag2 = 0.0;

			if ((0 <= angle_deg && angle_deg < 22.5) || (157.5 <= angle_deg && angle_deg < 180)) {
				mag1 = magnitude.at<float>(y, x - 1);
				mag2 = magnitude.at<float>(y, x + 1);
			}
			else if (22.5 <= angle_deg && angle_deg < 67.5) {
				mag1 = magnitude.at<float>(y - 1, x + 1);
				mag2 = magnitude.at<float>(y + 1, x - 1);
			}
			else if (67.5 <= angle_deg && angle_deg < 112.5) {
				mag1 = magnitude.at<float>(y - 1, x);
				mag2 = magnitude.at<float>(y + 1, x);
			}
			else if (112.5 <= angle_deg && angle_deg < 157.5) {
				mag1 = magnitude.at<float>(y - 1, x - 1);
				mag2 = magnitude.at<float>(y + 1, x + 1);
			}

			if (mag >= mag1 && mag >= mag2) {
				nms.at<uchar>(y, x) = static_cast<uchar>(mag);
			}
		}
	}

	return nms;
}

void edge_tracking(const Mat& nms, Mat& edges, Mat& visited, int x, int y, int lowThreshold) {
	if (x < 0 || x >= nms.cols || y < 0 || y >= nms.rows) {
		return;  // Out of bounds
	}
	if (visited.at<uchar>(y, x) == 1) {
		return;  // Already visited
	}
	if (nms.at<uchar>(y, x) < lowThreshold) {
		return;  // Not an edge
	}

	visited.at<uchar>(y, x) = 1;
	edges.at<uchar>(y, x) = 255;

	// Check 3 x 3 neighbours
	edge_tracking(nms, edges, visited, x - 1, y, lowThreshold); // Left
	edge_tracking(nms, edges, visited, x + 1, y, lowThreshold); // Right
	edge_tracking(nms, edges, visited, x, y - 1, lowThreshold); // Up
	edge_tracking(nms, edges, visited, x, y + 1, lowThreshold); // Down
	edge_tracking(nms, edges, visited, x - 1, y - 1, lowThreshold); // Top-left diagonal
	edge_tracking(nms, edges, visited, x + 1, y + 1, lowThreshold); // Bottom-right diagonal
	edge_tracking(nms, edges, visited, x - 1, y + 1, lowThreshold); // Bottom-left diagonal
	edge_tracking(nms, edges, visited, x + 1, y - 1, lowThreshold); // Top-right diagonal
}

void canny_manual() {
	Mat image = imread(IMG_PATH + "object2.png", IMREAD_GRAYSCALE);
	if (image.empty()) return;
	
	imshow("Original", image);

	Mat blurred;
	GaussianBlur(image, blurred, Size(5, 5), 0);

	imshow("Blurred", blurred);

	Mat grad_x, grad_y;
	Sobel(blurred, grad_x, CV_32F, 1, 0, 3);
	Sobel(blurred, grad_y, CV_32F, 0, 1, 3);

	Mat magnitude, angle;
	cartToPolar(grad_x, grad_y, magnitude, angle, true);

	imshow("Sobel", magnitude / 255.);

	Mat nms_result = apply_NMS(magnitude, angle);

	imshow("NMS", nms_result);

	Mat edges = Mat::zeros(nms_result.size(), CV_8UC1);
	Mat visited = Mat::zeros(nms_result.size(), CV_8UC1);

	int lowThreshold = 30;
	int highThreshold = 90;

	// mark strong edges
	for (int y = 1; y < nms_result.rows - 1; ++y) {
		for (int x = 1; x < nms_result.cols - 1; ++x) {
			if (nms_result.at<uchar>(y, x) >= highThreshold) {
				edges.at<uchar>(y, x) = 255;
				edge_tracking(nms_result, edges, visited, x, y, lowThreshold);
			}
		}
	}
	imshow("Tracking", edges);
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

void bbox() {
	Mat img = imread(IMG_PATH + "shapes.png", IMREAD_GRAYSCALE);
	if (img.empty()) return;

	Mat bin;
	threshold(img, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);
	imshow("bin", bin);

	Mat labels, stats, centroids;
	int cnt = connectedComponentsWithStats(bin, labels, stats, centroids);

	Mat dst;
	cvtColor(img, dst, COLOR_GRAY2BGR);

	for (int i = 1; i < cnt; i++) {
		int* p = stats.ptr<int>(i);
		if (p[4] < 20) continue;
		rectangle(dst, Rect(p[0], p[1], p[2], p[3]), Scalar(0, 255, 255), 2);
	}
	imshow("bbox", dst);
}

void contours() {
	Mat img = imread(IMG_PATH + "shapes.png", IMREAD_GRAYSCALE);
	if (img.empty()) return;

	vector<vector<Point>> contours;
	findContours(img, contours, RETR_LIST, CHAIN_APPROX_NONE);

	Mat dst;
	cvtColor(img, dst, COLOR_GRAY2BGR);

	for (int i = 0; i < contours.size(); i++) {
		Scalar c(rand() & 255, rand() & 255, rand() & 255);
		drawContours(dst, contours, i, c, 2);
	}
	imshow("dst", dst);
}

void contours_hierarchy() {
	Mat img = imread(IMG_PATH + "shapes.png", IMREAD_GRAYSCALE);
	if (img.empty()) return;

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(img, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

	Mat dst;
	cvtColor(img, dst, COLOR_GRAY2BGR);

	for (int i = 0; i  >= 0; i = hierarchy[i][0]) {
		Scalar c(rand() & 255, rand() & 255, rand() & 255);
		drawContours(dst, contours, i, c, -1, LINE_8, hierarchy);
	}
	imshow("dst", dst);
}

void bbox_with_label() {
	Mat img = imread(IMG_PATH + "shapes.png", IMREAD_COLOR);
	if (img.empty()) return;

	Mat img_gray;
	cvtColor(img, img_gray, COLOR_BGR2GRAY);

	Mat bin;
	threshold(img_gray, bin, 200, 255, THRESH_BINARY | THRESH_OTSU);

	vector<vector<Point>> contours;
	findContours(bin, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	string label;

	for (vector<Point>& pts : contours) {
		// skip if the area is smaller than...
		if (contourArea(pts) < 100) continue;

		// approximate Curve using Donglas-Peucker
		vector<Point> approx;
		approxPolyDP(pts, approx, arcLength(pts, true) * 0.02, true);

		int vtc = (int)approx.size();

		if (vtc == 3) {
			label = "TRI";
		}
		else if (vtc == 4) {
			label = "RECT";
		}
		else if (vtc > 4) {
			double len = arcLength(pts, true);
			double area = contourArea(pts);
			double ratio = 4. * CV_PI * area / (len * len);

			if (ratio > 0.8) {
				label = "CIRCLE";
			}
			else {
				label = "UNKNOWN";
			}
		}
		// draw bbox and label
		Rect rc = boundingRect(pts);
		rectangle(img, rc, Scalar(0, 0, 255), 1);
		putText(img, label, rc.tl(), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
	}
	imshow("bbox", img);
}
