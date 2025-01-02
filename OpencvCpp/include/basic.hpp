#pragma once

// computation
void bgr2gray();
void brightness_gray();
void brightness_manual();
void brightness_hsv();
void contrast();
void contrast2();
void mask_setTo();
void mask_copyTo();
void overlap();
void find_diff();
void mask_grayscale_bitwise();

// filter
void embossing();
void mean_blur();
void gaussian_blur();
void unsharp_mask();
void noise();
void remove_noise_gaussian();
void remove_noise_median();

// video
void camera_in();
void video_in();

// hist
void drawHist();
void compareBrightnessHist();
void compareContrastHist();