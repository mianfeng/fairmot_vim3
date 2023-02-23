// #pragma once

// #include "vnn_kdsbsr101ibnsbsr34simquantize.h"

// #include <opencv2/opencv.hpp>
// #include <Eigen/Dense>

// #include <string>
// #include <vector>

// typedef enum
// {
//     QUERY,
//     GALLERY
// }reid_status;

// class REID
// {
// public:
//     REID(const std::string& model);
//     ~REID();

//     int run(const cv::Mat& image, reid_status status);
//     int run_partial(const cv::Mat& image, reid_status status);
//     int run_multi(const cv::Mat& query_img, const std::vector<cv::Mat>& gallery_img, std::vector<float>& scores);
//     int run_partial_multi(const cv::Mat& query_img, const std::vector<cv::Mat>& gallery_img, std::vector<float>& scores);
//     float get_result();
//     void save_output_data();

//     reid_status status;
//     Eigen::MatrixXf output_query;
//     Eigen::MatrixXf output_gallery;

// private:
//     void run_pre(const cv::Mat& image);
//     void run_pre_partial(const cv::Mat& image);
//     void run_post();

//     vsi_nn_graph_t *graph;
//     vsi_nn_tensor_t *input_tensor;
//     uint32_t input_sz;
//     uint32_t input_stride;
//     vsi_nn_tensor_t *output_tensor;
//     uint32_t output_sz;

//     std::array<float, 3> scale;
//     std::array<float, 3> bias;
//     std::vector<uint8_t> input_uint8;
//     Eigen::MatrixXf output_mat;
// };
#pragma once

#include "vnn_kdsbsr101ibnsbsr34simquantize.h"
#include "datatype.h"

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include <string>
#include <vector>

class REID
{
public:
    REID(const std::string& model);
    ~REID();

    int run(const cv::Mat& bgr, FEATURE& feature);
    int runMulti(const std::vector<cv::Mat>& bgrs, FEATURESS& features);
    int getRectFeature(const cv::Mat& bgr, const cv::Rect& rect, FEATURE& feature);
    int getRectFeatureMulti(const cv::Mat& bgr, const std::vector<cv::Rect>& rects, FEATURESS& features);

    void save_output_data();

private:
    void run_pre(const cv::Mat& bgr);
    void run_post(FEATURE& feature);

    vsi_nn_graph_t *graph;
    vsi_nn_tensor_t *input_tensor;
    uint32_t input_sz;
    uint32_t input_stride;
    vsi_nn_tensor_t *output_tensor;
    uint32_t output_sz;

    std::array<float, 3> scale = {58.395, 57.120000000000005, 57.375};
    std::array<float, 3> bias = {123.675, 116.28, 103.53};

    std::vector<uint8_t> input_uint8;
};