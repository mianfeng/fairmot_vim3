#ifndef DATATYPE_H
#define DATATYPE_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#define REID_FEATURE_SIZE 512

class Object
{
  public:
    cv::Rect_<float> rect;
    int label;
    float prob;
};

using FEATURE = Eigen::Matrix<float, 1, REID_FEATURE_SIZE, Eigen::RowMajor>;
using FEATURESS = Eigen::Matrix<float, Eigen::Dynamic, REID_FEATURE_SIZE, Eigen::RowMajor>;
using DETECTBOX = Eigen::Matrix<float, 1, 4, Eigen::RowMajor>;
using DETECTBOXSS = Eigen::Matrix<float, -1, 4, Eigen::RowMajor>;
using RESULT = std::pair<int, Object>;
using RESULTS = std::vector<RESULT>;

class DETECTION {
  public:
    enum class DETECTBOX_IDX {IDX_X = 0, IDX_Y, IDX_W, IDX_H };

    Object object;
    float trackID;
    FEATURE feature;

    DETECTBOX to_xyah() const {
        //(centerx, centery, ration, h)
        DETECTBOX ret = {object.rect.x, object.rect.y, object.rect.width, object.rect.height};
        ret(0, int(DETECTBOX_IDX::IDX_X)) += (ret(0, int(DETECTBOX_IDX::IDX_W))*0.5);
        ret(0, int(DETECTBOX_IDX::IDX_Y)) += (ret(0, int(DETECTBOX_IDX::IDX_H))*0.5);
        ret(0, int(DETECTBOX_IDX::IDX_W)) /= ret(0, int(DETECTBOX_IDX::IDX_H));
        return ret;
    }
    DETECTBOX to_tlwh() const {
        //(leftx, topy, w, h)
        DETECTBOX ret = {object.rect.x, object.rect.y, object.rect.width, object.rect.height};
        return ret;
    }
};

using DETECTIONS = std::vector<DETECTION>;

//Kalmanfilter
using KAL_MEAN = Eigen::Matrix<float, 1, 8, Eigen::RowMajor>;
using KAL_COVA = Eigen::Matrix<float, 8, 8, Eigen::RowMajor>;
using KAL_HMEAN = Eigen::Matrix<float, 1, 4, Eigen::RowMajor>;
using KAL_HCOVA = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;
using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;

//linear_assignment:
using DYNAMICM = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;

//tracker:
using TRACKER_DATA = std::pair<int, FEATURESS>;
using MATCH_DATA = std::pair<int, int>;
class TRACHER_MATCHD{
  public:
    std::vector<MATCH_DATA> matches;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;
};

#endif //DEEPSORTDATATYPE_H