#include "opencv2/opencv.hpp"
#include <iostream>
//#include "basic.hpp"
//#include "transform.hpp"
#include "detection.hpp"

using namespace std;
using namespace cv;

int main() {
	hough_circles2();

	waitKey();
	destroyAllWindows();
	return 0;
}
