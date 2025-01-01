#ifdef __BASIC_H__

#else
#define __BASIC_H__

// video.cpp
void camera_in();
void video_in();

// features.cpp
void trackbar();
void mask_setTo();
void mask_copyTo();
void overlay();
void find_diff();
void mask_grayscale_bitwise();

// convert.cpp
void bgr2gray();
void brightness_gray();
void brightness_manual();
void brightness_hsv();
void contrast();
void contrast2();

// hist
void drawHist();
void compareBrightnessHist();
void compareContrastHist();

// filter
void embossing();
void mean_blur();
void gaussian_blur();

#endif
