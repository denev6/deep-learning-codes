#include "opencv2/opencv.hpp"
#include <iostream>
#include "detection.hpp"

using namespace std;
using namespace cv;

const string IMG_PATH = "C:\\Users\\admin\\Desktop\\deep-learning-codes\\OpencvCpp\\assets\\";
const string MODEL_PATH = "C:\\Users\\admin\\Desktop\\deep-learning-codes\\OpencvCpp\\models\\";

void template_matching() {
	Mat img = imread(IMG_PATH + "desk1.jpg");
	Mat template_img = imread(IMG_PATH + "template.jpg");
	if (img.empty() || template_img.empty()) return;

	// add noise
img = img + Scalar(50, 50, 50);
Mat noise(img.size(), CV_32SC3);
randn(noise, 0, 10);
add(img, noise, img, Mat(), CV_8UC3);

Mat dst, dst_normalized;
matchTemplate(img, template_img, dst, TM_CCOEFF_NORMED);
normalize(dst, dst_normalized, 0, 255, NORM_MINMAX, CV_8U);

double max_value;
Point max_loc;
minMaxLoc(dst, 0, &max_value, 0, &max_loc);

rectangle(img, Rect(max_loc.x, max_loc.y, template_img.cols, template_img.rows), Scalar(0, 0, 255), 2);

imshow("image with noises", img);
imshow("template", template_img);
imshow("correlation", dst_normalized);
}

void detect_face_haarcascade() {
	Mat img = imread(IMG_PATH + "p_face.jpg");
	if (img.empty()) return;

	CascadeClassifier face_classifier(MODEL_PATH + "haarcascade_frontalface_default.xml");
	CascadeClassifier eye_classifier(MODEL_PATH + "haarcascade_eye.xml");
	if (face_classifier.empty() || eye_classifier.empty()) return;

	vector<Rect> faces;
	face_classifier.detectMultiScale(img, faces);

	for (Rect face : faces) {
		rectangle(img, face, Scalar(0, 255, 0), 2);

		Mat face_ROI = img(face);
		vector<Rect> eyes;
		eye_classifier.detectMultiScale(face_ROI, eyes);

		for (Rect eye : eyes) {
			Point center(eye.x + eye.width / 2, eye.y + eye.height / 2);
			circle(face_ROI, center, eye.width / 2, Scalar(255, 0, 0), 2, LINE_AA);
		}
	}
	imshow("img", img);
}

void detect_people_hog() {
	Mat img = imread(IMG_PATH + "p_people.jpg");
	if (img.empty()) return;

	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	vector<Rect> detected;
	hog.detectMultiScale(img, detected);

	for (Rect rect : detected) {
		Scalar color = Scalar(rand() % 256, rand() % 256, rand() % 256);
		rectangle(img, rect, color, 2);
	}
	imshow("people", img);
}

void detect_QR() {
	Mat img = imread(IMG_PATH + "p_qr.png");
	if (img.empty()) return;

	QRCodeDetector detector;
	vector<Point> points;
	String info = detector.detectAndDecode(img, points);

	if (!info.empty()) {
		polylines(img, points, true, Scalar(0, 0, 255), 2);
		putText(img, info, Point(10, 30), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255), 2);
	}
	imshow("QR", img);
}

void corner_harris() {
	Mat img = imread(IMG_PATH + "object1.png", IMREAD_GRAYSCALE);
	if (img.empty()) return;

	Mat harris;
	// src, dst, block size, kernel size, k(.04 ~ .06)
	cornerHarris(img, harris, 3, 3, 0.04);

	Mat harris_norm;
	normalize(harris, harris_norm, 0, 255, NORM_MINMAX, CV_8U);

	Mat dst;
	cvtColor(img, dst, COLOR_GRAY2BGR);

	int threshold = 100;

	for (int row = 1; row < harris.rows - 1; row++) {
		for (int col = 1; col < harris.cols - 1; col++) {
			if (harris_norm.at<uchar>(row, col) > threshold) {
				if (harris.at<float>(row, col) > harris.at<float>(row - 1, col) &&
					harris.at<float>(row, col) > harris.at<float>(row + 1, col) &&
					harris.at<float>(row, col) > harris.at<float>(row, col - 1) &&
					harris.at<float>(row, col) > harris.at<float>(row, col + 1)) {
					circle(dst, Point(col, row), 3, Scalar(0, 0, 255), 2);
				} 
			}
		}
	}
	imshow("harris", harris_norm);
	imshow("corners", dst);
}

void corner_FAST() {
	Mat img = imread(IMG_PATH + "object1.png", IMREAD_GRAYSCALE);
	if (img.empty()) return;

	vector<KeyPoint> keypoints;
	int threshold = 120;
	FAST(img, keypoints, threshold, true);

	Mat dst;
	cvtColor(img, dst, COLOR_GRAY2BGR);

	for (KeyPoint keypoint : keypoints) {
		Point pt(cvRound(keypoint.pt.x), cvRound(keypoint.pt.y));
		circle(dst, pt, 3, Scalar(0, 0, 255), 2);
	}

	imshow("corner", dst);
}
