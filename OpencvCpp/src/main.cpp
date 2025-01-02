#include "opencv2/opencv.hpp"
#include <iostream>
#include "basic.hpp"
#include "transform.hpp"

using namespace std;
using namespace cv;

int main() {
	perspective_transform();

	waitKey();
	destroyAllWindows();
	return 0;
}
