
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;


const int max_value_H = 360 / 2;
const int max_value = 255;
const String window_detection_name = "Object Detection";
int low_H = 0, low_S = 64, low_V = 120;
int high_H = 72, high_S = 255, high_V = 253;


static void on_low_H_thresh_trackbar(int, void*)
{
    low_H = min(high_H - 1, low_H);
    setTrackbarPos("Low H", window_detection_name, low_H);
}
static void on_high_H_thresh_trackbar(int, void*)
{
    high_H = max(high_H, low_H + 1);
    setTrackbarPos("High H", window_detection_name, high_H);
}
static void on_low_S_thresh_trackbar(int, void*)
{
    low_S = min(high_S - 1, low_S);
    setTrackbarPos("Low S", window_detection_name, low_S);
}
static void on_high_S_thresh_trackbar(int, void*)
{
    high_S = max(high_S, low_S + 1);
    setTrackbarPos("High S", window_detection_name, high_S);
}
static void on_low_V_thresh_trackbar(int, void*)
{
    low_V = min(high_V - 1, low_V);
    setTrackbarPos("Low V", window_detection_name, low_V);
}
static void on_high_V_thresh_trackbar(int, void*)
{
    high_V = max(high_V, low_V + 1);
    setTrackbarPos("High V", window_detection_name, high_V);
}


const char* window_name = "Edge Map";

/*The operator simply sets to 1 all the pixels contained between
the low and high thresholds and to 0 all the other pixels.*/
//3
Mat binarization(Mat binarized) {
    cout << "in bin" << endl;
    /*inRange(inputhsv, Scalar(hLowThreshold, sLowThreshold, vLowThreshold),
        Scalar(hHighThreshold, sHighThreshold, vHighThreshold), mask);*/
    namedWindow(window_name, WINDOW_NORMAL);
    imshow(window_name, binarized);
    waitKey(0);

    return binarized;
}

void masking(Mat image, Mat binMask) {
    cout << "in masking" << endl;
    cv::Mat dstImage = cv::Mat::zeros(image.size(), image.type());
    //binMask = Mat::zeros(image.size(), CV_8UC1);
    //I assume you want to draw the circle at the center of your image, with a radius of 50
    cv::circle(binMask, cv::Point(binMask.cols / 2, binMask.rows / 2), 50, cv::Scalar(255, 0, 0), -1, 8, 0);

    //Now you can copy your source image to destination image with masking
    image.copyTo(dstImage, binMask);
    namedWindow(window_name, WINDOW_NORMAL);
    imshow(window_name, dstImage);
    waitKey(0);
}

void findContour(Mat binImage) {
    cout << "in contour\n";
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    Mat contours_image = Mat::zeros(binImage.size(), CV_8UC3); //initially there are no contours so all 0s

    findContours(binImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    // we need at least one contour to work
    if (contours.size() <= 0)
        cout << "oops\n";

    // find the biggest contour (let's suppose it's our hand)
    int biggest_contour_index = -1;
    double biggest_area = 0.0;

    for (int i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i], false);
        if (area > biggest_area) {
            biggest_area = area; //we hope this corresponds to the hand
            biggest_contour_index = i;
        }
    }
    cout << "biggest area" << biggest_area << endl;
    if (biggest_contour_index < 0)
        cout << "oops again\n";

    vector<Point> hull_points;
    vector<int> hull_ints;

    // for drawing the convex hull and for finding the bounding rectangle
    convexHull(Mat(contours[biggest_contour_index]), hull_points, true);

    // for finding the defects
    convexHull(Mat(contours[biggest_contour_index]), hull_ints, false);

    // we need at least 3 points to find the defects
    vector<Vec4i> defects;
    if (hull_ints.size() > 3)
        convexityDefects(Mat(contours[biggest_contour_index]), hull_ints, defects);
    else
        cout << "oops again haha\n";
    // we bound the convex hull
    Rect bounding_rectangle = boundingRect(Mat(hull_points));

    // we find the center of the bounding rectangle, this should approximately also be the center of the hand
    Point center_bounding_rect(
        (bounding_rectangle.tl().x + bounding_rectangle.br().x) / 2,
        (bounding_rectangle.tl().y + bounding_rectangle.br().y) / 2
    );

    // we separate the defects keeping only the ones of intrest
    vector<Point> start_points;
    vector<Point> far_points;

    for (int i = 0; i < defects.size(); i++) {
        start_points.push_back(contours[biggest_contour_index][defects[i].val[0]]);

        // filtering the far point based on the distance from the center of the bounding rectangle
        if (findPointsDistance(contours[biggest_contour_index][defects[i].val[2]], center_bounding_rect) < bounding_rectangle.height * BOUNDING_RECT_FINGER_SIZE_SCALING)
            far_points.push_back(contours[biggest_contour_index][defects[i].val[2]]);
    }

    // we compact them on their medians
    vector<Point> filtered_start_points = compactOnNeighborhoodMedian(start_points, bounding_rectangle.height * BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING);
    vector<Point> filtered_far_points = compactOnNeighborhoodMedian(far_points, bounding_rectangle.height * BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING);




}

/*After binarization the image resulted a bit noisy, because of false positives.
To clean the image and remove the false positives, an opening operator is applied,
with a 3x3 circular structuring element.
A dilation is also applied, just in case parts of the hand have been detached
after binarization (sometimes fingers are detached from the hand).*/
void performOpening(Mat binaryImage, int kernelShape, Point kernelSize) {
    int morph_size = 2;

    // Create structuring element
    //Mat element = getStructuringElement(
    //    MORPH_RECT,
    //    Size(2 * morph_size + 1,
    //        2 * morph_size + 1),
    //    Point(morph_size, morph_size));
    //Mat output;

    //// Closing
    //morphologyEx(binaryImage, output,
    //    MORPH_CLOSE, element,
    //    Point(-1, -1), 2);
    Mat structuringElement = getStructuringElement(kernelShape, kernelSize);
    morphologyEx(binaryImage, binaryImage, MORPH_OPEN, structuringElement);
    dilate(binaryImage, binaryImage, Mat(), Point(-1, -1), 3);
    Mat dst;
    //medianBlur(binaryImage, dst, 3);
    namedWindow(window_name, WINDOW_AUTOSIZE);
    imshow(window_name, binaryImage);
    waitKey(0);
}

/*calculates the low and high thresholds in each color channel*/
//2
void calculateThresholds(Mat handImage, Mat mask) {
    int offsetLowThreshold = 90;
    int offsetHighThreshold = 70;
            
    Scalar hsvMeans = mean(handImage); // has 3 values, mean in each color channel

    //hLowThreshold = hsvMeans[0] - offsetLowThreshold;
    //hHighThreshold = hsvMeans[0] + offsetHighThreshold;
    //
    //sLowThreshold = hsvMeans[1] - offsetLowThreshold;
    //sHighThreshold = hsvMeans[1] + offsetHighThreshold;

    //// the V channel shouldn't be used. By ignorint it, shadows on the hand wouldn't interfire with segmentation.
    //// Unfortunately there's a bug somewhere and not using the V channel causes some problem. This shouldn't be too hard to fix.
    //vLowThreshold = hsvMeans[2] - offsetLowThreshold;
    //vHighThreshold = hsvMeans[2] + offsetHighThreshold;
 
    
   /* vLowThreshold = 0;
    vHighThreshold = 255;*/
}



/*
1. convert to HSV
2. do thresholding in each color channel
3. morphological operations for removing false positives (false hands)
3.5 remove isolated white regions?
4. using a binary mask, do binary segmentation
5.
*/
int main(int argc, char** argv)
{
    //CommandLineParser parser(argc, argv, "{@input | fruits.jpg | input image}");
    //src = imread(samples::findFile(parser.get<String>("@input")), IMREAD_COLOR); // Load an image
  
 

    int choice;
    cout << "Enter 1 for object detection, 2 for segmentation: \n";
    cin >> choice;
    switch (choice) {
        case 1:
            cout << "nothign yet haha";
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

            Mat frame_HSV, frame_threshold;

            while (true) {
                Mat src = imread("24.jpg", IMREAD_COLOR);
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
            //Mat binarized = binarization(frame_threshold);
            performOpening(frame_threshold, MORPH_RECT, { 3, 3 });
            Mat mask = cv::imread("04.png", IMREAD_UNCHANGED);
            //mask = Mat::zeros(frame_threshold.size(), CV_8UC1);
            findContour(frame_threshold);
            /* cout << low_H<<endl;
             cout << high_H << endl;
             cout << low_S << endl;
             cout << high_S << endl;
             cout << low_V << endl;
             cout << high_V << endl;*/
            break;

    }
    
    return 0;
}