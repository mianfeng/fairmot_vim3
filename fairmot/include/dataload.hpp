#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>
#include <fstream>
#include <iomanip>

typedef enum
{
    CAMERA,
    VIDEO,
    IMAGE
} dataload_status;

class DATALOAD
{
public:
    DATALOAD(std::string video_name);
    // ~DATALOAD();

    int get_frame();
    cv::Mat frame;

private:
    cv::VideoCapture vp;
    std::vector<std::string> imagePathList;
    dataload_status status;
    std::vector<cv::String>::iterator img_idx;
    std::vector<cv::String>::iterator img_end;

    int d;
};
