#pragma once

#include <math.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "c_api.h"
#include "timer.hpp"
#include "datatype.h"

//default is 416，should be 32*n, need change anchor also
// #define YOLOV5_H 640
// #define YOLOV5_W 640

// struct Object
// {
//     cv::Rect_<float> rect;
//     int label;
//     float prob;
// };

class Yolov5
{
public:
    Yolov5(const std::string& model_path, const int model_h, const int model_w, const int cls_num);
    ~Yolov5();

    int run(const cv::Mat& bgr, const float prob_threshold, const float nms_threshold, std::vector<Object>& objects);
    void draw_result(cv::Mat& bgr, std::vector<Object>& objects);
    void cut_result(const cv::Mat& bgr, std::vector<Object>& objects, std::vector<cv::Mat>& person);

private:
    void run_pre(const cv::Mat& bgr);
    void run_post(const cv::Mat& bgr, const float prob_threshold, const float nms_threshold, std::vector<Object>& objects);

    const int model_h;
    const int model_w;
    const int cls_num;
    const float mean[3] = {0, 0, 0};
    const float scale[3] = {0.003921, 0.003921, 0.003921};

    graph_t graph;

    tensor_t input_tensor;
    std::vector<uint8_t> input_data;
    float input_scale = 0.f;
    int input_zero_point = 0;

    tensor_t p8_output;
    tensor_t p16_output;
    tensor_t p32_output;
    uint8_t* p8_data_u8;
    uint8_t* p16_data_u8;
    uint8_t* p32_data_u8;
    float p8_scale = 0.f;
    float p16_scale = 0.f;
    float p32_scale = 0.f;
    int p8_zero_point = 0;
    int p16_zero_point = 0;
    int p32_zero_point = 0;
    int p8_count;
    int p16_count;
    int p32_count;
    std::vector<float> p8_data;
    std::vector<float> p16_data;
    std::vector<float> p32_data;
};

static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"};
