#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

using namespace std;
using namespace cv;
using namespace dnn;


int main(int, char**){
    
    string file_path = ""; //specify the file path
    string class_name = "person"; //change to just string
    ifstream ifs(string(file_path + "object_detection_classes_coco.txt").c_str());
    string line;
    
    //read neural networks from the files
    auto net = readNet("frozen_inference_graph.pb", "graph.pbtxt", "TensorFlow");
    

    
    // CAN ADD GPU
    // net.setPreferableBackend(DNN_BACKEND_CUDA);
    // net.setPreferableTarget(DNN_TARGET_CUDA);
    
    
    // Set a min confidence score for the detections
    float min_confidente_score = 0.5;
    
    
        
        // load image
        Mat image = imread("26.jpg");
        
        int image_height = image.cols;
        int image_width = image.rows;
        
        
        auto start = getTickCount();
        
        //create a blob from the image
        Mat blob = blobFromImage(image, 1.0, Size(300,300), Scalar(127.5, 127.5, 127.5), true, false);
        // set the blob to be input of the neural network

        net.setInput(blob);
        
        //forward pass of the blob
        Mat output = net.forward();
        cout<< output.size();
        auto end = getTickCount();
        
        
        // matrix with all the detections
        Mat results(output.size[2], output.size[3], CV_32F, output.ptr<float>());
        
        //Run through all the predictions
        for (int i = 0; i < results.rows; i++){
            int class_id = int(results.at<float>(i, 1));
            int confidence = results.at<float>(i, 2);
            
            // Check if the detection is over the min threshold and then draw BB
            if (confidence > min_confidente_score){
                int bboxX = int(results.at<float>(i, 3) * image.cols);
                int bboxY = int(results.at<float>(i, 4) * image.rows);
                int bboxWidth = int(results.at<float>(i, 5) * image.cols - bboxX);
                int bboxHeight = int(results.at<float>(i, 6) * image.rows - bboxY);
                rectangle(image, Point(bboxX, bboxY), Point(bboxX + bboxWidth, bboxY + bboxHeight), Scalar(0, 0, 255), 2);
            }
        }
        
        imshow("image", image);
    
    return 0;
}
