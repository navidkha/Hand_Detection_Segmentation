
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <numeric>
using namespace cv;
using namespace std;


const int max_value_H = 360 / 2;
const int max_value = 255;
const String window_detection_name = "Object Detection";
int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value;
int thresh_low_H = 0; int thresh_low_S = 0; int thresh_low_V = 0; int thresh_high_H = 0;
int thresh_high_S = 0; int thresh_high_V = 0;


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

void masking(Mat image, Mat binMask) {
    cv::Mat dstImage = cv::Mat::zeros(image.size(), image.type());
    //binMask = Mat::zeros(image.size(), CV_8UC1);
    cv::circle(binMask, cv::Point(binMask.cols / 2, binMask.rows / 2), 50, cv::Scalar(255, 0, 0), -1, 8, 0);

    //Now you can copy your source image to destination image with masking
    image.copyTo(dstImage, binMask);
    namedWindow(window_name, WINDOW_NORMAL);
    imshow(window_name, dstImage);
    waitKey(0);
}

void findContour(Mat binImage) {
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

    //for (int i = 0; i < defects.size(); i++) {
    //    start_points.push_back(contours[biggest_contour_index][defects[i].val[0]]);

    //    // filtering the far point based on the distance from the center of the bounding rectangle
    //    if (findPointsDistance(contours[biggest_contour_index][defects[i].val[2]], center_bounding_rect) < bounding_rectangle.height * BOUNDING_RECT_FINGER_SIZE_SCALING)
    //        far_points.push_back(contours[biggest_contour_index][defects[i].val[2]]);
    //}

    //// we compact them on their medians
    //vector<Point> filtered_start_points = compactOnNeighborhoodMedian(start_points, bounding_rectangle.height * BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING);
    //vector<Point> filtered_far_points = compactOnNeighborhoodMedian(far_points, bounding_rectangle.height * BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING);




}

/*After binarization the image resulted a bit noisy, because of false positives.
To clean the image and remove the false positives, an opening operator is applied,
with a 3x3 circular structuring element.
A dilation is also applied, just in case parts of the hand have been detached
after binarization (sometimes fingers are detached from the hand).*/
void performOpening(Mat binaryImage, int kernelShape, Point kernelSize, Mat binMask) {
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
   /* namedWindow(window_name, WINDOW_AUTOSIZE);
    imshow(window_name, binaryImage);
    waitKey(0);*/
    masking(binaryImage, binMask);
}

void computeThresholds() {
    //choosing between mode, mean and median
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
    cout << total_low_H << endl;
    cout << (int)total_low_H << endl;
    /*cout << sizeof(low_H) / sizeof(low_H[0]);*/

    //MEAN
    thresh_low_H = total_low_H / 30;
    thresh_low_S = total_low_S / 30;
    thresh_low_V = total_low_V / 30;
    thresh_high_V = total_high_V / 30;
    thresh_high_S = total_high_S / 30;
    thresh_high_H = total_high_H / 30;

    //MEDIAN
    sort(low_H, low_H + (30));
    sort(low_S, low_S + (30));
    sort(low_V, low_V + (30));
    sort(high_H, high_H + (30));
    sort(high_S, high_S + (30)));
    sort(high_V, high_V + (30));
    int med_low_H = (low_H[30 / 2] + low_H[(30 / 2) - 1]) / 2;
    int med_low_S = (low_S[30 / 2] + low_S[(30 / 2) - 1]) / 2;
    int med_low_V = (low_V[30 / 2] + low_V[(30 / 2) - 1]) / 2;
    int med_high_H = (high_H[30 / 2] + high_H[(30 / 2) - 1]) / 2;
    int med_high_S = (high_S[30 / 2] + high_S[(30 / 2) - 1]) / 2;
    int med_high_V = (high_V[30 / 2] + high_V[(30 / 2) - 1]) / 2;

    //MODE
    //.
    //.
    //.

    


}


/*
1. convert to HSV
2. do thresholding in each color channel
3. morphological operations for removing false positives (false hands) AND highlighting the hands better
4. using a binary mask, select the hands --> color the result and copy it on the original image
4'. find contours
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
            Mat src = imread("30.jpg", IMREAD_COLOR);
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
        Mat mask = cv::imread("01.png", IMREAD_UNCHANGED);
        computeThresholds();
        //performOpening(frame_threshold, MORPH_RECT, { 3, 3 }, mask);

        //mask = Mat::zeros(frame_threshold.size(), CV_8UC1);
        //findContour(frame_threshold);

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