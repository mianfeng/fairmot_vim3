/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.

 */

/*
 * Copyright (c) 2021, OPEN AI LAB
 * Author: xwwang@openailab.com
 */

#include <math.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"
#include "tracker.h"
// #define CUT

static inline float sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static inline float intersection_area(const Object& a, const Object& b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left,
                                  int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j) {
        while (faceobjects[i].prob > p) i++;

        while (faceobjects[j].prob < p) j--;

        if (i <= j) {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects) {
    if (faceobjects.empty()) return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects,
                              std::vector<Object>& pick_objects,
                              std::vector<int>& picked, float nms_threshold) {
    picked.clear();

    const int n = faceobjects.size();
    pick_objects.resize(n);
    int k = 0;
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = faceobjects[i].rect.area();
        // fprintf(stderr, "areas[i] = %f \n",areas[i]);

        // }
        //     // fprintf(stderr, "LINE:%d\n",__LINE__);

        // for (int i = 0; i < n; i++)
        // {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            // fprintf(stderr, "inter_area = %f union_area = %f\n",
            // inter_area,union_area);
            if (inter_area / union_area > nms_threshold) keep = 0;
        }

        if (keep) {
            picked.push_back(i);
            pick_objects[k++] = faceobjects[i];
        }
    }
}

void get_input_data_uint8(cv::Mat& sample, uint8_t* input_data, int img_h,
                          int img_w, const float* mean, const float* scale,
                          float input_scale, int zero_point) {
    // cv::Mat sample = cv::imread(image_file, 1);
    cv::Mat img;
    if (sample.channels() == 1)
        cv::cvtColor(sample, img, cv::COLOR_GRAY2RGB);
    else
        cv::cvtColor(sample, img, cv::COLOR_BGR2RGB);

    /* letterbox process to support different letterbox size */
    float scale_letterbox;
    int resize_rows;
    int resize_cols;
    if ((img_h * 1.0 / img.rows) < (img_w * 1.0 / img.cols)) {
        scale_letterbox = img_h * 1.0 / img.rows;
    } else {
        scale_letterbox = img_w * 1.0 / img.cols;
    }
    resize_cols = int(scale_letterbox * img.cols);
    resize_rows = int(scale_letterbox * img.rows);

    cv::resize(img, img, cv::Size(resize_cols, resize_rows));
    img.convertTo(img, CV_32FC3);
    // Generate a gray image for letterbox using opencv
    cv::Mat img_new;
    int top = (img_h - resize_rows) / 2;
    int bot = (img_h - resize_rows + 1) / 2;
    int left = (img_w - resize_cols) / 2;
    int right = (img_w - resize_cols + 1) / 2;
    // Letterbox filling
    cv::copyMakeBorder(img, img_new, top, bot, left, right, cv::BORDER_CONSTANT,
                       cv::Scalar(0, 0, 0));

    /* resize process */
    // cv::resize(img, img, cv::Size(img_w, img_h));
    img_new.convertTo(img_new, CV_32FC3);
    float* img_data = (float*)img_new.data;

    /* nhwc to nchw */
    for (int h = 0; h < img_h; h++) {
        for (int w = 0; w < img_w; w++) {
            for (int c = 0; c < 3; c++) {
                int in_index = h * img_w * 3 + w * 3 + c;
                int out_index = c * img_h * img_w + h * img_w + w;
                float input_fp32 = (img_data[in_index] - mean[c]) * scale[c];

                /* quant to uint8 */
                int udata =
                    (round)(input_fp32 / input_scale + (float)zero_point);
                if (udata > 255)
                    udata = 255;
                else if (udata < 0)
                    udata = 0;

                input_data[out_index] = udata;
            }
        }
    }
}

static void generate_proposals(int stride, const float* feat,
                               float prob_threshold,
                               std::vector<Object>& objects) {
    static float anchors[18] = {10, 13, 16,  30,  33, 23,  30,  61,  62,
                                45, 59, 119, 116, 90, 156, 198, 373, 326};

    int anchor_num = 3;
    int feat_w = 608.0 / stride;
    int feat_h = 608.0 / stride;
    int cls_num = 80;
    int anchor_group = 0;
    if (stride == 8) anchor_group = 1;
    if (stride == 16) anchor_group = 2;
    if (stride == 32) anchor_group = 3;
    // printf("anchor_group:%d\n",anchor_group);
    for (int h = 0; h <= feat_h - 1; h++) {
        for (int w = 0; w <= feat_w - 1; w++) {
            for (int anchor = 0; anchor <= anchor_num - 1; anchor++) {
                int class_index = 0;
                float class_score = -FLT_MAX;
                int channel_size = feat_h * feat_w;
                for (int s = 0; s <= cls_num - 1; s++) {
                    int score_index = anchor * 85 * channel_size + feat_w * h +
                                      w + (s + 5) * channel_size;
                    float score = feat[score_index];
                    if (score > class_score) {
                        class_index = s;
                        class_score = score;
                    }
                }
                float box_score = feat[anchor * 85 * channel_size + feat_w * h +
                                       w + 4 * channel_size];
                float final_score = sigmoid(box_score) * sigmoid(class_score);
                if (final_score >= prob_threshold) {
                    int dx_index = anchor * 85 * channel_size + feat_w * h + w +
                                   0 * channel_size;
                    int dy_index = anchor * 85 * channel_size + feat_w * h + w +
                                   1 * channel_size;
                    int dw_index = anchor * 85 * channel_size + feat_w * h + w +
                                   2 * channel_size;
                    int dh_index = anchor * 85 * channel_size + feat_w * h + w +
                                   3 * channel_size;

                    float dx = sigmoid(feat[dx_index]);

                    float dy = sigmoid(feat[dy_index]);

                    float dw = feat[dw_index];
                    float dh = feat[dh_index];

                    float anchor_w =
                        anchors[(anchor_group - 1) * 6 + anchor * 2 + 0];
                    float anchor_h =
                        anchors[(anchor_group - 1) * 6 + anchor * 2 + 1];

                    float pred_x = (w + dx) * stride;
                    float pred_y = (h + dy) * stride;
                    float pred_w = exp(dw) * anchor_w;
                    float pred_h = exp(dh) * anchor_h;

                    float x0 = (pred_x - pred_w * 0.5f);
                    float y0 = (pred_y - pred_h * 0.5f);
                    float x1 = (pred_x + pred_w * 0.5f);
                    float y1 = (pred_y + pred_h * 0.5f);

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.label = class_index;
                    obj.prob = final_score;
                    objects.push_back(obj);
                }
            }
        }
    }
}
cv::Scalar GetColor(int idx) {
    idx = idx * 3;
    cv::Scalar color =
        cv::Scalar((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255);
    return color;
}
void VisualizeTrackResult(const cv::Mat& img, const MOTResult& results,
                          int time) {
    cv::Mat vis_img = img.clone();
    int im_h = img.rows;
    int im_w = img.cols;
    float text_scale = std::max(1, static_cast<int>(im_w / 1600.));
    float text_thickness = 2.;
    float line_thickness = std::max(1, static_cast<int>(im_w / 500.));

    std::ostringstream oss;
    oss << std::setiosflags(std::ios::fixed) << std::setprecision(4);
    //   oss << "frame: " << frame_id << " ";
    //   oss << "fps: " << fps << " ";
    oss << "num: " << results.size();
    std::string text = oss.str();
    fprintf(stderr, "num: %ld\n", results.size());
    cv::Point origin;
    origin.x = 0;
    origin.y = static_cast<int>(15 * text_scale);
    cv::putText(vis_img, text, origin, cv::FONT_HERSHEY_PLAIN, text_scale,
                (0, 0, 255), 2);

    for (int i = 0; i < results.size(); i++) {
        const int obj_id = results[i].ids;
        const float score = results[i].score;

        cv::Scalar color = GetColor(obj_id);

        cv::Point pt1 = cv::Point(results[i].rects.left, results[i].rects.top);
        cv::Point pt2 =
            cv::Point(results[i].rects.right, results[i].rects.bottom);
        cv::Point id_pt =
            cv::Point(results[i].rects.left, results[i].rects.top + 20);
        cv::Point score_pt =
            cv::Point(results[i].rects.left, results[i].rects.top - 10);
        cv::rectangle(vis_img, pt1, pt2, color, line_thickness);

        std::ostringstream idoss;
        idoss << std::setiosflags(std::ios::fixed) << std::setprecision(4);
        idoss << "class:" << obj_id;
        std::string id_text = idoss.str();

        cv::putText(vis_img, id_text, id_pt, cv::FONT_HERSHEY_PLAIN,
                    text_scale + 1, cv::Scalar(0, 0, 255), text_thickness);

        std::ostringstream soss;
        soss << std::setiosflags(std::ios::fixed) << std::setprecision(2);
        soss << score;
        std::string score_text = soss.str();

        cv::putText(vis_img, score_text, score_pt, cv::FONT_HERSHEY_PLAIN,
                    text_scale, cv::Scalar(0, 255, 255), text_thickness);
    }
    static std::string video_name = "./out/fairmot_timvx_out";
    static std::string video_path = "./out/fairmot_timvx_out(0).mp4";
    static int NUM = 0;
    // NUM = time/100;
    static cv::VideoWriter video_out(
        video_path, 828601953, 20, cv::Size(vis_img.cols, vis_img.rows), true);
#ifdef CUT
    if (!(time % 400) || time == 1) {
        std::cout << video_path << "\n";
        video_path =
            video_name + "(" + std::__cxx11::to_string(NUM++) + ")" + ".mp4";
        cv::VideoWriter video_out1(video_path, 828601953, 30,
                                   cv::Size(vis_img.cols, vis_img.rows), true);
        video_out = video_out1;
        video_out1.release();
    }
#endif

    if (!video_out.isOpened())
        fprintf(stderr, "fairmot_timvx_out.mp4 can't create");
    if (!vis_img.empty()) {
        video_out.write(vis_img);
    }
}
static void draw_objects_video(const cv::Mat& bgr,
                               const std::vector<Object>& objects, int fourcc,
                               double fps, int frame_width, int frame_height,
                               int time) {
    //  fprintf(stderr,"fps=%f \n",fps);

    static const char* class_names[] = {"person"};

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i = i + 1) {
        const Object& obj = objects[i];

        // fprintf(stderr, "%2d: %3.0f%%, [%4.0f, %4.0f, %4.0f, %4.0f], %s\n",
        // obj.label, obj.prob * 100, obj.rect.x,
        //         obj.rect.y, obj.rect.area(), obj.rect.y + obj.rect.height,
        //         class_names[obj.label]);

        cv::rectangle(image, obj.rect, cv::Scalar(100, 255, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 30;
        cv::Size label_size =
            cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0) y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(
            image,
            cv::Rect(cv::Point(x, y),
                     cv::Size(label_size.width, label_size.height + baseLine)),
            cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    static std::string video_name = "./out/fairmot_timvx_out";
    static std::string video_path = "./out/fairmot_timvx_out(0).mp4";
    static int NUM = 0;
    NUM = time / 100;
    static cv::VideoWriter video_out(video_path, fourcc, 16,
                                     cv::Size(image.cols, image.rows), true);
    if (!(time % 150) || time == 1) {
        std::cout << video_path << "\n";
        video_path =
            video_name + "(" + std::__cxx11::to_string(NUM) + ")" + ".mp4";
        cv::VideoWriter video_out1(video_path, fourcc, 16,
                                   cv::Size(image.cols, image.rows), true);
        video_out = video_out1;
        video_out1.release();
    }

    if (!video_out.isOpened())
        fprintf(stderr, "fairmot_timvx_out.mp4 can't create");
    if (!image.empty()) {
        video_out.write(image);
    }
}
void show_usage() {
    fprintf(stderr,
            "[Usage]:  [-h]\n    [-m model_file] [-i image_file]  [-n ues "
            "webcam]\n");
    fprintf(stderr,
            "    [-v]:  video_name,video:xxx.mp4/avi\n    [-m]: model_file\n   "
            " [-n]: ues webcam\n");
}
void FlowStatistic(const MOTResult& results, const int frame_id,
                   const int secs_interval, const bool do_entrance_counting,
                   const int video_fps, const Rect entrance,
                   std::set<int>* id_set, std::set<int>* interval_id_set,
                   std::vector<int>* in_id_list, std::vector<int>* out_id_list,
                   std::map<int, std::vector<float> >* prev_center,
                   std::vector<std::string>* records) {
    if (frame_id == 0) interval_id_set->clear();

    if (do_entrance_counting) {
        // Count in and out number:
        // Use horizontal center line as the entrance just for simplification.
        // If a person located in the above the horizontal center line
        // at the previous frame and is in the below the line at the current
        // frame, the in number is increased by one. If a person was in the
        // below the horizontal center line at the previous frame and locates in
        // the below the line at the current frame, the out number is increased
        // by one.
        // TODO(qianhui): if the entrance is not the horizontal center line,
        // the counting method should be optimized.

        float entrance_y = entrance.top;
        for (const auto& result : results) {
            float center_x = (result.rects.left + result.rects.right) / 2;
            float center_y = (result.rects.top + result.rects.bottom) / 2;
            int ids = result.ids;
            std::map<int, std::vector<float> >::iterator iter;
            iter = prev_center->find(ids);
            if (iter != prev_center->end()) {
                if (iter->second[1] <= entrance_y && center_y > entrance_y) {
                    in_id_list->push_back(ids);
                }
                if (iter->second[1] >= entrance_y && center_y < entrance_y) {
                    out_id_list->push_back(ids);
                }
                (*prev_center)[ids][0] = center_x;
                (*prev_center)[ids][1] = center_y;
            } else {
                prev_center->insert(std::pair<int, std::vector<float> >(
                    ids, {center_x, center_y}));
            }
        }
    }

    // Count totol number, number at a manual-setting interval
    for (const auto& result : results) {
        id_set->insert(result.ids);
        interval_id_set->insert(result.ids);
    }
}
int video_input(const char* video_file, const char* model_file,
                float prob_threshold, int state) {
    // //set_log_level(log_level::LOG_DEBUG);
    // std::vector<std::string> imagePathList;
    // cv::VideoCapture cap;
    int img_h = 320;
    int img_w = 576;
    int img_c = 3;
    // std::vector<cv::String>::iterator img_idx;
    // std::vector<cv::String>::iterator img_end;
    // cv::Mat frame;
    // MOTResult result;
    // const float mean[3] = {0, 0, 0};
    // const float scale[3] = {0.003921, 0.003921, 0.003921};

    // if (state == 1)
    // {
    //     cap.open(video_file);
    //     if (!cap.isOpened())
    //     {
    //         fprintf(stderr, "video %s open failed\n", video_file);
    //         return -1;
    //     }
    //     fprintf(stderr, "video %s open \n", video_file);
    // }
    // else
    // {
    //     cv::glob(video_file, imagePathList);
    //     img_idx = imagePathList.begin();
    //     img_end = imagePathList.end();
    //     frame = cv::imread(*img_idx);
    //     if (frame.empty())
    //     {
    //         fprintf(stderr, "Reading file image was failed.\n");
    //         return -1;
    //     }
    // }

    /* set runtime options */
    struct options opt;
    opt.num_thread = 6;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_UINT8;
    opt.affinity = 0;

    /* inital tengine */
    if (init_tengine() != 0) {
        fprintf(stderr, "Initial tengine failed.\n");
        return -1;
    }
    fprintf(stderr, "tengine-lite library version: %s\n",
            get_tengine_version());

    /* create VeriSilicon TIM-VX backend */
    context_t timvx_context = create_context("timvx", 1);
    int rtt = set_context_device(timvx_context, "TIMVX", nullptr, 0);
    if (0 > rtt) {
        fprintf(stderr, " add_context_device VSI DEVICE failed.\n");
        return -1;
    }
    fprintf(stderr, " add_context_device VSI DEVICE ok.\n");
    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(timvx_context, "tengine", model_file);
    if (graph == nullptr) {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    int img_size = img_h * img_w * img_c;
    int dims[] = {1, 3, img_h, img_w};
    std::vector<uint8_t> input_data(img_size);
    /* get input tensor */
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);

    float input_scale = 0.f;
    int input_zero_point = 0;
    get_tensor_quant_param(input_tensor, &input_scale, &input_zero_point, 1);

    if (input_tensor == nullptr) {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0) {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor, input_data.data(), img_size) < 0) {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0) {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }
    fprintf(stderr, "Net ok.\n");
    // int frame_width = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    // int frame_height = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    // int frame_num = cap.get(7);
    // double fps = cap.get(cv::CAP_PROP_FPS);
    // int fourcc = cap.get(cv::CAP_PROP_FOURCC);
    // fprintf(stdout, "video w: %d  h: %d  fps: %.2f\r\n",
    //         frame_width, frame_height, fps);
    cv::Mat frame;
    int time = 0;
    while (true) {
        fprintf(stderr, "%d\n", time++);
        // if (state)
        //     cap.read(frame);
        // else
        //     frame = cv::imread(*img_idx++);
        // // if (frame.empty())
        // if (frame.empty() || time == 1600)
        // {
        //     fprintf(stderr, "run img empty\n");
        //     return -1;
        // }
        /* prepare process input data, set the data mem to input tensor */
        // get_input_data_uint8(frame, input_data.data(), img_h, img_w, mean,
        //                      scale, input_scale, input_zero_point);

        /* run graph */
        // double min_time = DBL_MAX;
        // double max_time = DBL_MIN;
        double total_time = 0.;
        double start = get_current_time();
        if (run_graph(graph, 1) < 0) {
            fprintf(stderr, "Run graph failed\n");
            return -1;
        }
        double end = get_current_time();
        double cur = end - start;
        // total_time += cur;
        // min_time = std::min(min_time, cur);
        // max_time = std::max(max_time, cur);
        fprintf(stdout, "Run graph cost %.2fms.\n", cur);
        // /* get output tensor */
        // tensor_t size_output = get_graph_output_tensor(graph, 0, 0);
        // tensor_t offset_output = get_graph_output_tensor(graph, 1, 0);
        // tensor_t feat_output = get_graph_output_tensor(graph, 2, 0);
        // tensor_t heatmap_output = get_graph_output_tensor(graph, 3, 0);
        // // if(heatmap_output)
        // tensor_t pool_output = get_graph_output_tensor(graph, 4, 0);
        // //  fprintf(stderr, " heatmap_output :%s\n"
        // //  ,get_tensor_name(pool_output));

        // float offset_scale = 0.f;
        // float feat_scale = 0.f;
        // float heatmap_scale = 0.f;
        // float size_scale = 0.f;
        // float pool_scale = 0.f;
        // int offset_zero_point = 0;
        // int feat_zero_point = 0;
        // int heatmap_zero_point = 0;
        // int size_zero_point = 0;
        // int pool_zero_point = 0;
        // int heatmap_count, pool_count, size_count;
        // int flag;
        // get_tensor_quant_param(feat_output, &feat_scale, &feat_zero_point,
        // 1); get_tensor_quant_param(offset_output, &offset_scale,
        // &offset_zero_point,
        //                        1);
        // get_tensor_quant_param(heatmap_output, &heatmap_scale,
        //                        &heatmap_zero_point, 1);
        // get_tensor_quant_param(size_output, &size_scale, &size_zero_point,
        // 1); get_tensor_quant_param(pool_output, &pool_scale,
        // &pool_zero_point, 1);

        // heatmap_count = pool_count =
        //     get_tensor_buffer_size(pool_output) / sizeof(uint8_t);

        // /* dequant output data */
        // uint8_t* offset_data_u8 = (uint8_t*)get_tensor_buffer(offset_output);
        // uint8_t* feat_data_u8 = (uint8_t*)get_tensor_buffer(feat_output);
        // uint8_t* heatmap_data_u8 =
        // (uint8_t*)get_tensor_buffer(heatmap_output); uint8_t* size_data_u8 =
        // (uint8_t*)get_tensor_buffer(size_output); uint8_t* pool_data_u8 =
        // (uint8_t*)get_tensor_buffer(pool_output);

        // std::vector<float> heatmap_data(heatmap_count);
        // std::vector<Object> objects;
        // std::vector<Object> objects_pick;
        // for (size_t i = 0; i < heatmap_count; i++) {
        //     // float heatmap_data = ((float)heatmap_data_u8[i] -
        //     // (float)heatmap_zero_point) * heatmap_scale;
        //     float heatmap_data =
        //         ((float)pool_data_u8[i] - (float)pool_zero_point) *
        //         pool_scale;
        //     float pool_data =
        //         ((float)pool_data_u8[i] - (float)pool_zero_point) *
        //         pool_scale;
        //     // printf("heatmap_data:%f, feat_data:%f\n",
        //     //     heatmap_data, pool_data);
        //     if (heatmap_data == pool_data && heatmap_data >= prob_threshold)
        //     {
        //         Object obj;
        //         obj.idx = i;
        //         obj.prob = heatmap_data;
        //         objects.push_back(std::move(obj));
        //     }
        // }
        // qsort_descent_inplace(objects);

        // /* postprocess */
        // // box and feat
        // /*draw the result */
        // float scale_letterbox;
        // int resize_rows;
        // int resize_cols;
        // std::vector<int> picked;

        // if ((img_h * 1.0 / frame.rows) < (img_w * 1.0 / frame.cols)) {
        //     scale_letterbox = img_h * 1.0 / frame.rows;
        // } else {
        //     scale_letterbox = img_w * 1.0 / frame.cols;
        // }
        // resize_cols = int(scale_letterbox * frame.cols);
        // resize_rows = int(scale_letterbox * frame.rows);

        // int tmp_h = (img_h - resize_rows) / 2;
        // int tmp_w = (img_w - resize_cols) / 2;

        // float ratio_x = (float)frame.rows / resize_rows;
        // float ratio_y = (float)frame.cols / resize_cols;
        // for (auto& obj : objects) {
        //     float ys = obj.idx / 144;
        //     float xs = obj.idx % 144;
        //     float offset_x = ((float)offset_data_u8[obj.idx * 2 + 0] -
        //                       (float)offset_zero_point) *
        //                      offset_scale;
        //     float offset_y = ((float)offset_data_u8[obj.idx * 2 + 1] -
        //                       (float)offset_zero_point) *
        //                      offset_scale;
        //     xs += offset_x;
        //     ys += offset_y;

        //     float size_0 = ((float)size_data_u8[obj.idx * 4 + 0] -
        //                     (float)size_zero_point) *
        //                    size_scale;
        //     float size_1 = ((float)size_data_u8[obj.idx * 4 + 1] -
        //                     (float)size_zero_point) *
        //                    size_scale;
        //     float size_2 = ((float)size_data_u8[obj.idx * 4 + 2] -
        //                     (float)size_zero_point) *
        //                    size_scale;
        //     float size_3 = ((float)size_data_u8[obj.idx * 4 + 3] -
        //                     (float)size_zero_point) *
        //                    size_scale;
        //     float x1 = xs - size_0;
        //     float y1 = ys - size_1;
        //     float x2 = xs + size_2;
        //     float y2 = ys + size_3;
        //     x1 = (x1 * 4 - tmp_w) * ratio_x;
        //     y1 = (y1 * 4 - tmp_h) * ratio_y;
        //     x2 = (x2 * 4 - tmp_w) * ratio_x;
        //     y2 = (y2 * 4 - tmp_h) * ratio_y;

        //     x1 = std::max(std::min(x1, (float)(frame.cols - 1)), 0.f);
        //     y1 = std::max(std::min(y1, (float)(frame.rows - 1)), 0.f);
        //     x2 = std::max(std::min(x2, (float)(frame.cols - 1)), 0.f);
        //     y2 = std::max(std::min(y2, (float)(frame.rows - 1)), 0.f);

        //     obj.rect.x = x1;
        //     obj.rect.y = y1;
        //     obj.rect.width = x2 - x1;
        //     obj.rect.height = y2 - y1;
        //     for (size_t i = 0; i < 128; i++) {
        //         // obj.feat(0, i) = ((float)feat_data_u8[obj.idx * 128 + i] -
        //         // (float)feat_zero_point) * feat_scale;
        //         obj.feat.at<float>(0, i) =
        //             ((float)feat_data_u8[obj.idx * 128 + i] -
        //              (float)feat_zero_point) *
        //             feat_scale;
        //     }
        //     cv::normalize(obj.feat, obj.feat, cv::NORM_L2);
        // }
        // nms_sorted_bboxes(objects, objects_pick, picked, 0.4);
        // int count = picked.size();
        // fprintf(stderr, "pick_size: %d obj_size: %ld\n", count,
        // objects.size()); objects_pick.resize(count);

        // std::vector<Track> tracks;
        // JDETracker::instance()->update(objects_pick, &tracks);
        // MOTResult result;
        // if (tracks.size() == 0) {
        //     MOTTrack mot_track;
        //     Rect ret = {0, 0, 0, 0};

        //     mot_track.ids = 1;
        //     mot_track.score = 0;
        //     mot_track.rects = ret;
        //     // result.push_back(mot_track);
        // } else {
        //     std::vector<Track>::iterator titer;
        //     for (titer = tracks.begin(); titer != tracks.end(); ++titer) {
        //         if (titer->score < prob_threshold) {
        //             continue;
        //         } else {
        //             float w = titer->ltrb[2] - titer->ltrb[0];
        //             float h = titer->ltrb[3] - titer->ltrb[1];
        //             bool vertical = w / h > 1.6;
        //             float area = w * h;
        //             if (area > 200 && !vertical) {
        //                 MOTTrack mot_track;
        //                 Rect ret = {titer->ltrb[0], titer->ltrb[1],
        //                             titer->ltrb[2], titer->ltrb[3]};
        //                 mot_track.rects = ret;
        //                 mot_track.score = titer->score;
        //                 mot_track.ids = titer->id;
        //                 result.push_back(mot_track);
        //             }
        //         }
        //     }
        // }
        // double start_time = get_current_time();
        // VisualizeTrackResult(frame, result, time);
        // // fprintf(stderr, "[L] %d\n",__LINE__);
        // // result.clear();
        // // draw_objects_video(frame,
        // // objects_pick,828601953,fps,frame_width,frame_height,time);
        // end = get_current_time();
        // fprintf(stdout, "draw cost %.2fms.\n", end - start_time);

        // cur = end - start;
        // fprintf(stdout, "all cost %.2fms.\n", cur);
    }

    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();
}
int main(int argc, char* argv[]) {
    const char* model_file = nullptr;
    const char* image_file = nullptr;
    const char* video_file = nullptr;

    int camera_num = 0;
    float find_thresh = 0.5;
    int state = 1;
    int res;
    while ((res = getopt(argc, argv, "m:r:t:hv:n:y:")) != -1) {
        switch (res) {
            case 'm':
                model_file = optarg;
                break;
            case 'v':
                video_file = optarg;
                break;
            case 'n':
                camera_num = std::strtoul(optarg, nullptr, 10);
                video_file = "net";
                break;
            case 'y':
                find_thresh = std::strtod(optarg, nullptr);
                break;
            case 'h':
                show_usage();
                return 0;
            default:
                break;
        }
    }

    /* check files */
    if (nullptr == model_file) {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (nullptr != video_file) {
        if (video_file == "net") {
            switch (camera_num) {
                case 0:
                    video_file =
                        "rtsp://admin:abc123456@192.168.0.80:554/cam/"
                        "realmonitor?channel=1&subtype=0";
                    break;
                case 1:
                    video_file =
                        "rtsp://admin:abc123456@192.168.0.81:554/cam/"
                        "realmonitor?channel=1&subtype=0";
                    break;
                case 2:
                    video_file =
                        "rtsp://admin:abc123456@192.168.0.82:554/cam/"
                        "realmonitor?channel=1&subtype=0";
                    break;
                case 3:
                    video_file = "./test_image/img1/";
                    state = 0;  // img_mode
                    break;
                default:
                    break;
            }
        }
        fprintf(stderr, "state = %d thresh= %f\n", state, find_thresh);
        if (video_input(video_file, model_file, find_thresh, state) < 0)
            return -1;
    } else {
        fprintf(stderr, "Error: video or image file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file) || !check_file_exist(video_file))
        return -1;
}