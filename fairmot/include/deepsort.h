#ifndef DEEPSORT_H
#define DEEPSORT_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "yolov5s.hpp"
#include "reid.hpp"
// #include "light_reid.hpp"

#include "tracker_d.h"
#include "datatype.h"
#include <vector>

class DeepSort {
  public:    
    DeepSort(const std::string& detectModelPath, const std::string& reidModelPath);
    void detect(const cv::Mat& bgr, RESULTS& results);
    void drawDetection(cv::Mat& bgr, const RESULTS& results);

  private:
    std::shared_ptr<Tracker> mTracker;
    std::shared_ptr<Yolov5> mDetect;
    std::shared_ptr<REID> mREID;
};

#endif  //deepsort.h
