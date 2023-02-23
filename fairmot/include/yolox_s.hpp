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

//default is 416，should be 32*n, need change anchor also
#define YOLOX_S_H 640
#define YOLOX_S_W 640

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class Yolox_s
{
public:
    Yolox_s(const std::string& model, float thresh);
    ~Yolox_s();

    int run(const cv::Mat& image);
    void draw_result(cv::Mat& img);
    int cut_result(cv::Mat& img);

    std::vector<Object> objects;
    std::vector<cv::Mat> person;

    float prob_threshold;//0.3

    static const char* class_names[80];

private:
    void run_pre();
    void run_post();

    int img_h = YOLOX_S_H;
    int img_w = YOLOX_S_W;
    // const float mean[3] = {255.f * 0.485f, 255.f * 0.456, 255.f * 0.406f};
    // const float scale[3] = {1 / (255.f * 0.229f), 1 / (255.f * 0.224f), 1 / (255.f * 0.225f)};
    const float mean[3] = {0.f, 0.f, 0.f};
    const float scale[3] = {1.f, 1.f, 1.f};

    graph_t graph;

    tensor_t input_tensor;
    std::vector<uint8_t> input_data;
    float input_scale = 0.f;
    int input_zero_point = 0;

    tensor_t p8_output;
    uint8_t* output_u8;
    float output_scale = 0.f;
    int output_zero_point = 0;
    int output_size;
    std::vector<float> p8_data;

    cv::Mat sample;

};
