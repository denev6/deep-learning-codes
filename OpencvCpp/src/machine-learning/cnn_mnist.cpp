#include "opencv2/opencv.hpp"
#include <iostream>
#include "machine-learning.hpp"

using namespace std;
using namespace cv;
using namespace cv::dnn;

const string MODEL_PATH = "C:\\Users\\admin\\Desktop\\deep-learning-codes\\OpencvCpp\\models\\";

void _on_mouse_mnist(int event, int x, int y, int flags, void* userdata);

void draw_mnist_cnn() {
	Net net = readNet(MODEL_PATH + "mnist_cnn.pb");
	if (net.empty()) return;

	Mat palette = Mat::zeros(400, 400, CV_8UC1);
	imshow("palette", palette);
	setMouseCallback("palette", _on_mouse_mnist, (void*)&palette);

	while (true) {
		int c = waitKey();

		if (c == 27) break; // ESC
		else if (c == 13) { // Enter
			Mat inputBlob = blobFromImage(palette, 1 / 255.f, Size(28, 28));
			net.setInput(inputBlob);
			Mat output = net.forward();

			double maxVal;
			Point maxLoc;
			minMaxLoc(output, NULL, &maxVal, NULL, &maxLoc);
			int digit = maxLoc.x;

			cout << digit << format("({%d})%", maxVal * 100) << endl;

			palette.setTo(0);
			imshow("prediction", palette);
		}
	}
}

Point _ptPrev(-1, -1);

void _on_mouse_mnist(int event, int x, int y, int flags, void* userdata) {
	Mat img = *(Mat*)userdata;

	if (event == EVENT_LBUTTONDOWN) {
		_ptPrev = Point(x, y);
	}
	else if (event == EVENT_LBUTTONUP) {
		_ptPrev = Point(-1, -1);
	}
	else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON)) {
		line(img, _ptPrev, Point(x, y), Scalar::all(255), 40, LINE_AA, 0);
		_ptPrev = Point(x, y);
		imshow("palette", img);
	}
}
