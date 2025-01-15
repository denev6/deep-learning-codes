#include "opencv2/opencv.hpp"
#include <iostream>
#include "machine-learning.hpp"

using namespace cv;
using namespace cv::ml;
using namespace std;

const string IMG_PATH = "C:\\Users\\admin\\Desktop\\deep-learning-codes\\OpencvCpp\\assets\\";

Ptr<KNearest> train_knn();

void draw_mnist_knn() {
	Ptr<KNearest> knn = train_knn();
	if (knn.empty()) return;

	Mat palette = Mat::zeros(400, 400, CV_8U);
	imshow("palette", palette);
	setMouseCallback("palette", on_mouse_mnist, (void*)&palette);

	while (true) {
		int c = waitKey();

		if (c == 27) break; // ESC
		else if (c == 13) { // Enter
			Mat img_resize, img_float, img_flatten, res;

			resize(palette, img_resize, Size(20, 20), 0, 0, INTER_AREA);
			img_resize.convertTo(img_float, CV_32F);
			img_flatten = img_float.reshape(1, 1);

			knn->findNearest(img_flatten, 3, res);
			cout << cvRound(res.at<float>(0, 0)) << endl;

			palette.setTo(0);
		}
	}
}

Ptr<KNearest> train_knn() {
	Mat mnist = imread(IMG_PATH + "p_digits.png", IMREAD_GRAYSCALE);
	if (mnist.empty()) return 0;

	Mat train_images, train_labels;

	// format trainset
	for (int j = 0; j < 50; j++) {
		for (int i = 0; i < 100; i++) {
			Mat roi, roi_float, roi_flatten;
			roi = mnist(Rect(i * 20, j * 20, 20, 20));
			roi.convertTo(roi_float, CV_32F);
			roi_flatten = roi_float.reshape(1, 1);

			train_images.push_back(roi_flatten);
			train_labels.push_back(j / 5);
		}
	}
	// train KNN
	Ptr<KNearest> knn = KNearest::create();
	knn->train(train_images, ROW_SAMPLE, train_labels);

	return knn;
}
