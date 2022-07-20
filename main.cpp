#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <string>
#include <sstream>

using namespace std;
using namespace cv;


const size_t inWidth = 300;
const size_t inHeight = 300;
const float WHRatio = inWidth / (float)inHeight;
const char* classNames[] = { "background","hand" };// According to the needs of this category



int main() {

    //load images
    vector<cv::String> fn;
    glob("rgb/*.jpg", fn, false);
    size_t count = fn.size(); //number of jpg files in images folder
    Mat frame;
    Mat blob;
    Mat output;
    

    //load model
    String weights = "frozen_inference_graph.pb";
    String prototxt = "graph.pbtxt";
    dnn::Net net = cv::dnn::readNetFromTensorflow(weights, prototxt);
    
    
    for (size_t i=0; i<count; i++){
        frame = imread(fn[i]);
        Size frame_size = frame.size();

        Size cropSize;
        if (frame_size.width / (float)frame_size.height > WHRatio)
        {
            cropSize = Size(static_cast<int>(frame_size.height * WHRatio),
                            frame_size.height);
        }
        else
        {
            cropSize = Size(frame_size.width,
                            static_cast<int>(frame_size.width / WHRatio));
        }

        Rect crop(Point((frame_size.width - cropSize.width) / 2,
                        (frame_size.height - cropSize.height) / 2),
                  cropSize);


        blob = cv::dnn::blobFromImage(frame, 1. / 255, Size(300, 300));
        //cout << "blob size: " << blob.size << endl;

        net.setInput(blob);
        output = net.forward();
        //cout << "output size: " << output.size << endl;

        Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

        frame = frame(crop);
        
        //threshold the network output according to the confidence measurement
        float confidenceThreshold = 0.30;
        for (int j = 0; j < detectionMat.rows; j++)
        {
            float confidence = detectionMat.at<float>(j, 2);

            if (confidence > confidenceThreshold)
            {
                size_t objectClass = (size_t)(detectionMat.at<float>(j, 1));

                int xLeftBottom = static_cast<int>(detectionMat.at<float>(j, 3) * frame.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(j, 4) * frame.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(j, 5) * frame.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(j, 6) * frame.rows);

                ostringstream ss;
                ss << confidence;
                String conf(ss.str());

                Rect object((int)xLeftBottom, (int)yLeftBottom,
                            (int)(xRightTop - xLeftBottom),
                            (int)(yRightTop - yLeftBottom));

                rectangle(frame, object, Scalar(0, 255, 0), 2);
                String label = String(classNames[objectClass]) + ": " + conf;
                int baseLine = 0;
                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
                                      Size(labelSize.width, labelSize.height + baseLine)),
                                      Scalar(0, 255, 0), FILLED);
                putText(frame, label, Point(xLeftBottom, yLeftBottom),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
            }
        }
        
        string file_name = "image" + to_string(i) + ".jpg";
        imwrite(file_name, frame);

    }
    return 0;
}
