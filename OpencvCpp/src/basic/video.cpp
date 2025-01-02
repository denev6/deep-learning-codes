#include "opencv2/opencv.hpp"
#include <iostream>
#include "basic.hpp"

using namespace std;
using namespace cv;

const string IMG_PATH = "C:\\Users\\admin\\Desktop\\deep-learning-codes\\OpencvCpp\\assets\\";

void camera_in() {
	VideoCapture cap(0);

	if (!cap.isOpened()) {
		return;
	}

	Mat frame, inversed;
	while (true) {
		cap >> frame;
		if (frame.empty()) break;

		inversed = ~frame;
		imshow("inversed", inversed);

		if (waitKey(10) == 27) { // ESC
			break;
		}
	}
	destroyAllWindows();
}

void video_in() {
	VideoCapture cap(IMG_PATH + "sample.mp4");

	if (!cap.isOpened()) {
		return;
	}
	double fps = cap.get(CAP_PROP_FPS);
	cout << "FPS: " << fps << endl;

	Mat frame, inversed;
	while (true) {
		cap >> frame;
		if (frame.empty()) break;

		inversed = ~frame;
		imshow("inversed", inversed);

		if (waitKey(10) == 27) { // ESC
			break;
		}
	}
	destroyAllWindows();
}