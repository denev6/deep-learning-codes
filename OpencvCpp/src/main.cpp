#include "opencv2/opencv.hpp"
#include <iostream>
//#include "basic.hpp"
//#include "transform.hpp"
#include "detection.hpp"
//#include "machine-learning.hpp"

using namespace std;
using namespace cv;

int main() {
	detect_people_hog();
	waitKey();
	return 0;
}
