
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


int main( int argc, const char** argv)
{
    
    Mat img = imread("01.jpg");
    Mat grey;
    Mat blur;
    Mat canny;
    
    int lowThreshold = 130;
    const int max_lowThreshold = 1000;
    const int ratio = 3;
    const int kernel_size = 3;
    
    //show basic image
    namedWindow("image", WINDOW_AUTOSIZE);
    imshow("image", img);
    
    
    //image processing part
    
    //grey scale
    cvtColor(img, grey, COLOR_BGR2GRAY);
    namedWindow("grey scale", WINDOW_AUTOSIZE);
    imshow("grey scale", grey);
    
    //Gaussian blur
    GaussianBlur(grey, blur, Size(3,3), 0);
    namedWindow("gaussian blur", WINDOW_AUTOSIZE);
    imshow("gaussian blur", blur);
    
    //Canny edge detector
    namedWindow( "canny", WINDOW_AUTOSIZE );
    createTrackbar( "Min Threshold:", "canny", &lowThreshold, max_lowThreshold);
    Canny(blur, canny, lowThreshold, lowThreshold*ratio, kernel_size);
    namedWindow("canny", WINDOW_AUTOSIZE);
    imshow("canny", canny);
    waitKey();
    
    return 0;
}
