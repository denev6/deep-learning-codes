#include "opencv2/opencv.hpp"
#include <iostream>
#include "machine-learning.hpp"

using namespace std;
using namespace cv;
using namespace cv::dnn;

const string MODEL_PATH = "C:\\Users\\admin\\Desktop\\deep-learning-codes\\OpencvCpp\\models\\";

void draw_mnist_cnn() {
	Net net = readNet(MODEL_PATH + "cnn.onnx");
	if (net.empty()) return;

	Mat palette = Mat::zeros(400, 400, CV_8UC1);
	imshow("palette", palette);
	setMouseCallback("palette", on_mouse_mnist, (void*)&palette);

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
