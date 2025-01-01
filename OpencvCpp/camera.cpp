#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;

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