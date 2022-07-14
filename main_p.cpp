
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <numeric>
using namespace cv;
using namespace std;

const int max_value_H = 360 / 2;
const int max_value = 255;
const String window_segmentation_name = "Segmentation";
const char* window_name = "Edge Map";
const String  window_detection_name = "the thing";
int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value;
int thresh_low_H = 0; int thresh_low_S = 0; int thresh_low_V = 0; int thresh_high_H = 0;
int thresh_high_S = 0; int thresh_high_V = 0;
int med_low_H, med_low_S, med_low_V, med_high_H, med_high_S, med_high_V;
RNG rng(12345);


static void on_low_H_thresh_trackbar(int, void*)
{
    low_H = min(high_H - 1, low_H);
    setTrackbarPos("Low H", window_segmentation_name, low_H);
}
static void on_high_H_thresh_trackbar(int, void*)
{
    high_H = max(high_H, low_H + 1);
    setTrackbarPos("High H", window_segmentation_name, high_H);
}
static void on_low_S_thresh_trackbar(int, void*)
{
    low_S = min(high_S - 1, low_S);
    setTrackbarPos("Low S", window_segmentation_name, low_S);
}
static void on_high_S_thresh_trackbar(int, void*)
{
    high_S = max(high_S, low_S + 1);
    setTrackbarPos("High S", window_segmentation_name, high_S);
}
static void on_low_V_thresh_trackbar(int, void*)
{
    low_V = min(high_V - 1, low_V);
    setTrackbarPos("Low V", window_segmentation_name, low_V);
}
static void on_high_V_thresh_trackbar(int, void*)
{
    high_V = max(high_V, low_V + 1);
    setTrackbarPos("High V", window_segmentation_name, high_V);
}

int getMaxAreaContourId(vector <vector<cv::Point>> contours) {
    double maxArea = 0;
    double newArea = 0;
    int maxAreaContourId = -1;
    for (int j = 0; j < contours.size(); j++) {
        newArea = cv::contourArea(contours.at(j));
        cout << newArea << " area of the contour " << j << endl;
        if (newArea > maxArea) {
            maxArea = newArea;
            maxAreaContourId = j;

        } // End if
    } // End for
    return maxAreaContourId;
}

vector<vector<cv::Point>>  getContoursIDs(vector<vector<cv::Point>> contours) {
    vector<vector<cv::Point>> selectedContours;
    double contourArea;
    int contThreshold = 4000;
    for (int j = 0; j < contours.size(); j++) {
        contourArea = cv::contourArea(contours.at(j));
        if (contourArea >= contThreshold) {
            selectedContours.push_back(contours.at(j)); //add contour ID to the vector
            cout << typeid(contours.at(j)).name() << endl;
            cout << contourArea << " is bigger than threshold" << contThreshold << endl;
        }
    }
    return selectedContours;
}

void pixelAccuracy(Mat contourIM, Mat binMask) {
    vector<int> acc;
    Mat gray_contours, bin_contours;
    //part 1: pixel accuracy for the hand class
    //the contour image and the mask have the same size

    cvtColor(contourIM, gray_contours, cv::COLOR_RGB2GRAY);
    int thresh = 138;
    threshold(gray_contours, bin_contours, thresh, 255, 3);


    //im_bw = threshold(gray_contours, thresh, 255, cv2.THRESH_BINARY)[1]
  /*  for (int i = 0; i < contourIM.rows; i++) {
        for (int j = 0; j < contourIM.rows; j++) {

        }

    }*/

    //part 2: pixel accuracy for the non-hand class
}
Mat findContour(Mat binImage, Mat RGB_src) {
    vector<vector<Point>> contours; //Detected contours. Each contour is stored as a vector of points
    vector<Vec4i> hierarchy; //has as many elements as the number of contours
    Scalar color;
    vector<vector<Point>> final_contours;
    Mat dst = Mat::zeros(binImage.rows, binImage.cols, CV_8UC3); //

    //find all contours in the binary image
    findContours(binImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    //find the contour ID corresponding to the one with the max area (hopefully the hand!)
    final_contours = getContoursIDs(contours);

    //draw the contour with the max area only (segmenting one hand only)
    for (int i = 0; i < final_contours.size(); i++) {
        color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));//chooses a color in RGB, with random values in each color channel
        drawContours(RGB_src, final_contours, (int)i, color, FILLED, 8);
    }
    return RGB_src;
}

Mat performClosing(Mat binaryImage, int kernelShape, Point kernelSize) {
    Mat structuringElement = getStructuringElement(kernelShape, kernelSize);
    morphologyEx(binaryImage, binaryImage, MORPH_OPEN, structuringElement);
    Mat kernel = getStructuringElement(MORPH_RECT, { 7, 7 });
    dilate(binaryImage, binaryImage, kernel, Point(-1, -1), 3);
    return binaryImage;
}

Mat performDialation(Mat image) {
    Mat structuringElement = getStructuringElement(MORPH_RECT, { 7, 7 });
    dilate(image, image, structuringElement, Point(-1, -1), 4);
    return image;
}

Mat performErosion(Mat originalImage) {
    //if Mat(), a 3 x 3 rectangular structuring element is used
    Mat structuringElement = getStructuringElement(MORPH_RECT, { 7, 7 });
    erode(originalImage, originalImage, structuringElement, Point(-1, -1), 4);
    return originalImage;
}

void computeThresholds() {
    int low_H[] = { 10, 5, 1, 1, 0, 0, 0, 9, 3, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0 ,9, 0 ,3, 0, 4 };
    int low_S[] = { 76, 20, 31, 75, 14, 40, 25, 10, 0, 24, 60, 75, 47, 40, 39, 75, 55, 63, 55, 64, 65, 68, 105, 72, 66, 62, 61, 62, 62, 121, };
    int low_V[] = { 116, 181, 194, 187, 189, 123, 130, 115, 141, 119, 137, 56, 65, 117, 105, 94, 114, 55, 113, 130, 50, 107, 185, 129, 91, 210, 129, 181, 211, 122 };
    int high_H[] = { 17, 12, 12, 11, 13, 14, 14, 16, 16, 13, 15, 18, 12, 12, 14, 17, 17, 15, 15, 13, 13, 9, 18, 25, 180, 100, 180, 175, 38, 19 };
    int high_S[] = { 183, 117, 113, 116, 114, 186, 184, 168, 148, 126, 148, 150, 117, 116, 144, 236, 135, 155, 138, 102, 104, 66, 66, 173, 104, 96, 145, 81, 47, 171, };
    int high_V[] = { 253, 255, 253, 213, 255, 199, 192, 255, 235, 222, 225, 146, 189, 207, 173, 204, 180, 154, 185, 211, 171, 231, 155, 255, 255, 255, 255, 255, 255, 157 };
    int total_low_H = accumulate(begin(low_H), end(low_H), 0, plus<int>());
    int total_low_S = accumulate(begin(low_S), end(low_S), 0, plus<int>());
    int total_low_V = accumulate(begin(low_V), end(low_V), 0, plus<int>());
    int total_high_V = accumulate(begin(high_V), end(high_V), 0, plus<int>());
    int total_high_S = accumulate(begin(high_S), end(high_S), 0, plus<int>());
    int total_high_H = accumulate(begin(high_H), end(high_H), 0, plus<int>());
    /*cout << sizeof(low_H) / sizeof(low_H[0]);*/

    //MEAN
    thresh_low_H = total_low_H / 30;
    thresh_low_S = total_low_S / 30;
    thresh_low_V = total_low_V / 30;
    thresh_high_V = total_high_V / 30;
    thresh_high_S = total_high_S / 30;
    thresh_high_H = total_high_H / 30;

    //MEDIAN (**been tested. not a good idea**)
    /*sort(low_H, low_H + (sizeof(low_H) / sizeof(low_H[0])));
    sort(low_S, low_S + (sizeof(low_S) / sizeof(low_S[0])));
    sort(low_V, low_V + (sizeof(low_V) / sizeof(low_V[0])));
    sort(high_H, high_H + (sizeof(high_H) / sizeof(high_H[0])));
    sort(high_S, high_S + (sizeof(high_S) / sizeof(high_S[0])));
    sort(high_V, high_V + (sizeof(high_V) / sizeof(high_V[0])));
    int med_low_H = (low_H[30 / 2] + low_H[(30 / 2) - 1]) / 2;
    int med_low_S = (low_S[30 / 2] + low_S[(30 / 2) - 1]) / 2;
    int med_low_V = (low_V[30 / 2] + low_V[(30 / 2) - 1]) / 2;
    int med_high_H = (high_H[30 / 2] + high_H[(30 / 2) - 1]) / 2;
    int med_high_S = (high_S[30 / 2] + high_S[(30 / 2) - 1]) / 2;
    int med_high_V = (high_V[30 / 2] + high_V[(30 / 2) - 1]) / 2;*/


}

/*
1.      convert to HSV
2.      Erosion (removing artifacts like the chessboard) followed by Dilation (recovering hands)
2.      do thresholding in each color channel
3.      morphological operations for closing small holes in big objects (which are hopefully hands)
4.      using a binary mask, select the hands --> color the result and copy it on the original image
4'.     find contours (returning contours having an area larger than a threshold)
*/

int main(int argc, char** argv)
{
    int choice;
    Mat src, frame_HSV, frame_threshold, output_morph, output_contour, final_image, eroded, dialated;

    cout << "Enter 1 for segmentation, 2 for thresholding trackbars: \n";
    cin >> choice;
    switch (choice) {
    case 1:
        src = imread("17.jpg", IMREAD_COLOR);
        if (src.empty())
        {
            std::cout << "Could not open or find the image!\n" << std::endl;
            std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
            //return -1;
            break;
        }
        //step 0: erosion on hsv image
        cvtColor(src, frame_HSV, COLOR_BGR2HSV); //convert RGB to HSV
        eroded = performErosion(frame_HSV); //rmv small white noise, detach connected objects
        dialated = performDialation(frame_HSV); //now that the noise is gone, increase the target object area

        //step 1: thresholding  
        computeThresholds(); //update the global threshold variables
        //cvtColor(src, frame_HSV, COLOR_BGR2HSV); //convert RGB to HSV
        inRange(eroded, Scalar(thresh_low_H, thresh_low_S, thresh_low_V), Scalar(thresh_high_H, thresh_high_S, thresh_high_V), frame_threshold); //inRange thresholding
        //step 2: closing followed by dialtion
        output_morph = performClosing(frame_threshold, MORPH_RECT, { 3, 3 });
        //step 3: find contours
        output_contour = findContour(output_morph, src);
        // copy only non-zero pixels from your image to original image


        namedWindow(window_detection_name, WINDOW_NORMAL);
        imshow(window_detection_name, output_contour);
        waitKey(0);
        break;

    case 2:
        namedWindow(window_detection_name, WINDOW_NORMAL);
        // Trackbars to set thresholds for HSV values
        createTrackbar("Low H", window_detection_name, &low_H, max_value_H, on_low_H_thresh_trackbar);
        createTrackbar("High H", window_detection_name, &high_H, max_value_H, on_high_H_thresh_trackbar);
        createTrackbar("Low S", window_detection_name, &low_S, max_value, on_low_S_thresh_trackbar);
        createTrackbar("High S", window_detection_name, &high_S, max_value, on_high_S_thresh_trackbar);
        createTrackbar("Low V", window_detection_name, &low_V, max_value, on_low_V_thresh_trackbar);
        createTrackbar("High V", window_detection_name, &high_V, max_value, on_high_V_thresh_trackbar);

        while (true) {
            src = imread("24.jpg", IMREAD_COLOR);
            if (src.empty())
            {
                std::cout << "Could not open or find the image!\n" << std::endl;
                std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
                //return -1;
                break;
            }
            // Convert from BGR to HSV colorspace
            cvtColor(src, frame_HSV, COLOR_BGR2HSV);
            // Detect the object based on HSV Range Values
            inRange(frame_HSV, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), frame_threshold);
            Mat resized_frame;
            //resize(frame_threshold, resized_frame, Size(), 0.5, 0.5);
   /*         int width = 1200;
            int height = 800;*/
            //resizeWindow(int(width * (height - 80) / height), height - 80);
            imshow(window_detection_name, frame_threshold);
            char key = (char)waitKey(30);
            if (key == 'q' || key == 27)
            {
                break;
            }
        }
        Mat mask = cv::imread("04.png", IMREAD_UNCHANGED);
        break;
    }
    return 0;
}