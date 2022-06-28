#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <vector>

using namespace cv;
RNG rng(12345);

Mat src, src_gray;
Mat dst, detected_edges, opening;

int lowThreshold = 0;
int highThreshold = 350;
const int max_lowThreshold = 1000;
const int max_highThreshold = 1000;
const int kernel_size = 3;
const char* window_name = "Edge Map";


static void CannyThreshold(int, void*)
{
    blur( src_gray, detected_edges, Size(3,3) );
    Canny( detected_edges, detected_edges, lowThreshold, highThreshold, kernel_size );
    imshow( window_name, detected_edges );
    Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  
    // Opening
    morphologyEx(detected_edges, opening, MORPH_DILATE, element, Point(-1, -1), 1);
    imshow("Opening", opening);
}


int main( int argc, char** argv )
{
  src = imread("01.jpg");
  cvtColor( src, src_gray, COLOR_BGR2GRAY );
  imshow("source",src);
  namedWindow( window_name, WINDOW_AUTOSIZE );
  createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );
  createTrackbar( "Max Threshold:", window_name, &highThreshold, max_highThreshold, CannyThreshold );
  CannyThreshold(0, 0);
    
    std::vector<std::vector<Point> > contours;
    std::vector<Vec4i> hierarchy;
    findContours(opening, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat drawing = Mat::zeros( src.size(), CV_8UC3 );
    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        drawContours( drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0 );
    }
        imshow( "Contours", drawing );
  waitKey(0);
  return 0;
}
