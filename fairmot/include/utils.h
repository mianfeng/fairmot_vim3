//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <ctime>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

// #include "include/tracker.h"

struct Object
{
    int idx;
    int label = 0;
    float prob;
    cv::Rect_<float> rect;
    // Eigen::Matrix<float, 1, 128, Eigen::RowMajor> feat;
    cv::Mat feat = cv::Mat(1, 128, CV_32FC1);
};

// namespace PaddleDetection {



struct Rect {
  float left;
  float top;
  float right;
  float bottom;
};

struct MOTTrack {
  int ids;
  float score;
  Rect rects;
  int class_id = -1;
};

typedef std::vector<MOTTrack> MOTResult;

// }  // namespace PaddleDetection