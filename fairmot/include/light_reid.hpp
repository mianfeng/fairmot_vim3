#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "c_api.h"
#include "timer.hpp"
#include "datatype.h"

#include <string>
#include <vector>

class LIGHT_REID
{
public:
    LIGHT_REID(const std::string& model);
    ~LIGHT_REID();

    int run(const cv::Mat& bgr, FEATURE& feature);
    int getRectFeature(const cv::Mat& bgr, const cv::Rect& rect, FEATURE& feature);

private:
    void run_pre(const cv::Mat& bgr);
    void run_post(FEATURE& feature);

    float mean[3] = {123.67f, 116.28f, 103.53f};
    float scale[3] = {0.017125f, 0.017507f, 0.017429f};

    int img_h = 256;
    int img_w = 128;

    graph_t graph;

    tensor_t input_tensor;
    std::vector<float> input_data;
    // float input_scale = 0.f;
    // int input_zero_point = 0;

    tensor_t output_tensor;
    float* output = NULL;
    // std::vector<float> output_data;
    int output_size;
    // float output_scale = 0.f;
    // int output_zero_point = 0;
};