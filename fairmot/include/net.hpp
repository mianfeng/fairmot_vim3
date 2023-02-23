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
#include <Eigen/Core>
#include <Eigen/Dense>

#include "tengine/c_api.h"
#include "timer.hpp"
#include "utils.h"

// #define Net_H 320
// #define Net_W 576
#define BENCHMARK
#define TIM_VX

// struct Object
// {
//     int idx;
//     int label = 0;
//     float prob;
//     cv::Rect_<float> rect;
//     Eigen::Matrix<float, 1, 128, Eigen::RowMajor> feat;
// };

class Net
{
public:
    Net(const std::string& model_path, const int model_h, const int model_w);
    ~Net();

    int run(const cv::Mat& bgr, const float prob_threshold, std::vector<Object>& objects);
    void draw_result(cv::Mat& bgr, std::vector<Object>& objects);
    void cut_result(const cv::Mat& bgr, std::vector<Object>& objects, std::vector<cv::Mat>& person);

private:
    void run_pre(const cv::Mat& bgr);
    void run_post(const cv::Mat& bgr, const float prob_threshold, std::vector<Object>& objects);

    int max_per_img = 500;

    const int model_h;
    const int model_w;
    const float mean[3] = {0, 0, 0};
    const float scale[3] = {0.003921, 0.003921, 0.003921};

    graph_t graph;

    tensor_t input_tensor;
#ifdef TIM_VX
    std::vector<uint8_t> input_data;
    float input_scale = 0.f;
    int input_zero_point = 0;
#else
    std::vector<float> input_data;
#endif
    tensor_t offset_output;
    tensor_t feat_output;
    tensor_t heatmap_output;
    tensor_t size_output;
    tensor_t pool_output;
#ifdef TIM_VX
    uint8_t* offset_data_u8;
    uint8_t* feat_data_u8;
    uint8_t* heatmap_data_u8;
    uint8_t* size_data_u8;
    uint8_t* pool_data_u8;
    float offset_scale = 0.f;
    float feat_scale = 0.f;
    float heatmap_scale = 0.f;
    float size_scale = 0.f;
    float pool_scale = 0.f;
    int offset_zero_point = 0;
    int feat_zero_point = 0;
    int heatmap_zero_point = 0;
    int size_zero_point = 0;
    int pool_zero_point = 0;
    int heatmap_count;
    int pool_count ;
#else
    float* offset_data_u8;
    float* feat_data_u8;
    float* heatmap_data_u8;
    float* size_data_u8;
    float* pool_data_u8;
    int heatmap_count;
#endif
};

static const char* class_names[] = {"person"};
