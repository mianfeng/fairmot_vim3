#ifndef TRACKER_H
#define TRACKER_H


#include <vector>

#include "kalmanfilter.h"
#include "track.h"
#include "datatype.h"

using namespace std;

class NearNeighborDisMetric;

class Tracker
{
public:
    shared_ptr<NearNeighborDisMetric> metric;
    float max_iou_distance;
    int max_age;
    int n_init;

    shared_ptr<KalmanFilter> kf;

    int _next_idx;
public:
    std::vector<Track> tracks;
    Tracker(/*NearNeighborDisMetric* metric,*/
    		float max_cosine_distance, int nn_budget,
            float max_iou_distance = 0.7,
            int max_age = 60, int n_init=3);
    void predict();
    void update(const DETECTIONS& detections);
    typedef DYNAMICM (Tracker::* GATED_METRIC_FUNC)(
            std::vector<Track>& tracks,
            const DETECTIONS& dets,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices);
private:    
    void _match(const DETECTIONS& detections, TRACHER_MATCHD& res);
    void _initiate_track(const DETECTION& detection);
public:
    DYNAMICM gated_matric(
            std::vector<Track>& tracks,
            const DETECTIONS& dets,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices);
    DYNAMICM iou_cost(
            std::vector<Track>& tracks,
            const DETECTIONS& dets,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices);
    Eigen::VectorXf iou(DETECTBOX& bbox,
            DETECTBOXSS &candidates);
};

#endif // TRACKER_H
