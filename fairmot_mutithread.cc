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
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "common.h"
#include "demuxing.h"
#include "encoder.h"
#include "ionplayer.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"
#include "tracker.h"
#include "utils.h"

// #define CUT
int img_h = 320;
int img_w = 576;
int img_c = 3;

struct FairmotGraph {
    graph_t graph;
    std::vector<uint8_t> input_data;
    float input_scale = 0.f;
    int input_zero_point = 0;
    MOTResult result;
    float prob_threshold;
};
struct VideoData {
    int size;
    void* VideoPtr;
    cv::Mat frame;
    cv::Mat outFrame;
};

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
cv::Mat VisualizeTrackResult(const cv::Mat& img, const MOTResult& results) {
    cv::Mat vis_img = img.clone();
    //   vis_img.data
    //   vis_img.size.
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

    for (long unsigned int i = 0; i < results.size(); i++) {
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
                    text_scale + 0.5, cv::Scalar(0, 0, 255), text_thickness);

        std::ostringstream soss;
        soss << std::setiosflags(std::ios::fixed) << std::setprecision(2);
        soss << score;
        std::string score_text = soss.str();

        cv::putText(vis_img, score_text, score_pt, cv::FONT_HERSHEY_PLAIN,
                    text_scale, cv::Scalar(0, 255, 255), text_thickness);
    }
    cv::Mat yuvImg;
    cvtColor(vis_img, yuvImg, cv::COLOR_RGB2YUV_YV12);
    return yuvImg;
}

void show_usage() {
    fprintf(stderr,
            "[Usage]:  [-h]\n    [-m model_file] [-i image_file]  [-n ues "
            "webcam]\n");
    fprintf(stderr,
            "    [-v]:  video_name,video:xxx.mp4/avi\n    [-m]: model_file\n   "
            " [-n]: ues webcam\n");
}

int initFairmotGraph(FairmotGraph* fairmotgraph, const char* model_file) {
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
    fairmotgraph->graph = create_graph(timvx_context, "tengine", model_file);
    if (fairmotgraph->graph == nullptr) {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    int img_size = img_h * img_w * img_c;
    int dims[] = {1, 3, img_h, img_w};
    fairmotgraph->input_data.resize(img_size);
    /* get input tensor */
    tensor_t input_tensor = get_graph_input_tensor(fairmotgraph->graph, 0, 0);

    get_tensor_quant_param(input_tensor, &(fairmotgraph->input_scale),
                           &(fairmotgraph->input_zero_point), 1);

    if (input_tensor == nullptr) {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0) {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor, fairmotgraph->input_data.data(),
                          img_size) < 0) {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(fairmotgraph->graph, opt) < 0) {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }
    fprintf(stderr, "Net ok.\n");
    return 0;
}
int outFairmotGraph(FairmotGraph* fairmotgraph, int frame_height,
                    int frame_width) {
    /* get output tensor */
    tensor_t size_output = get_graph_output_tensor(fairmotgraph->graph, 0, 0);
    tensor_t offset_output = get_graph_output_tensor(fairmotgraph->graph, 1, 0);
    tensor_t feat_output = get_graph_output_tensor(fairmotgraph->graph, 2, 0);
    tensor_t heatmap_output =
        get_graph_output_tensor(fairmotgraph->graph, 3, 0);
    // if(heatmap_output)
    tensor_t pool_output = get_graph_output_tensor(fairmotgraph->graph, 4, 0);
    //  fprintf(stderr, " heatmap_output :%s\n" ,get_tensor_name(pool_output));

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
    int heatmap_count, pool_count, size_count;
    int flag;
    get_tensor_quant_param(feat_output, &feat_scale, &feat_zero_point, 1);
    get_tensor_quant_param(offset_output, &offset_scale, &offset_zero_point, 1);
    get_tensor_quant_param(heatmap_output, &heatmap_scale, &heatmap_zero_point,
                           1);
    get_tensor_quant_param(size_output, &size_scale, &size_zero_point, 1);
    get_tensor_quant_param(pool_output, &pool_scale, &pool_zero_point, 1);

    heatmap_count = pool_count =
        get_tensor_buffer_size(pool_output) / sizeof(uint8_t);

    /* dequant output data */
    uint8_t* offset_data_u8 = (uint8_t*)get_tensor_buffer(offset_output);
    uint8_t* feat_data_u8 = (uint8_t*)get_tensor_buffer(feat_output);
    uint8_t* heatmap_data_u8 = (uint8_t*)get_tensor_buffer(heatmap_output);
    uint8_t* size_data_u8 = (uint8_t*)get_tensor_buffer(size_output);
    uint8_t* pool_data_u8 = (uint8_t*)get_tensor_buffer(pool_output);

    std::vector<float> heatmap_data(heatmap_count);
    std::vector<Object> objects;
    std::vector<Object> objects_pick;
    for (size_t i = 0; i < heatmap_count; i++) {
        // float heatmap_data = ((float)heatmap_data_u8[i] -
        // (float)heatmap_zero_point) * heatmap_scale;
        float heatmap_data =
            ((float)pool_data_u8[i] - (float)pool_zero_point) * pool_scale;
        float pool_data =
            ((float)pool_data_u8[i] - (float)pool_zero_point) * pool_scale;
        // printf("heatmap_data:%f, feat_data:%f\n",
        //     heatmap_data, pool_data);
        if (heatmap_data == pool_data &&
            heatmap_data >= fairmotgraph->prob_threshold) {
            Object obj;
            obj.idx = i;
            obj.prob = heatmap_data;
            objects.push_back(std::move(obj));
        }
    }
    qsort_descent_inplace(objects);

    /* postprocess */
    // box and feat
    /*draw the result */
    float scale_letterbox;
    int resize_rows;
    int resize_cols;
    std::vector<int> picked;

    if ((img_h * 1.0 / frame_height) < (img_w * 1.0 / frame_width)) {
        scale_letterbox = img_h * 1.0 / frame_height;
    } else {
        scale_letterbox = img_w * 1.0 / frame_width;
    }
    resize_cols = int(scale_letterbox * frame_width);
    resize_rows = int(scale_letterbox * frame_height);

    int tmp_h = (img_h - resize_rows) / 2;
    int tmp_w = (img_w - resize_cols) / 2;

    float ratio_x = (float)frame_height / resize_rows;
    float ratio_y = (float)frame_width / resize_cols;
    for (auto& obj : objects) {
        float ys = obj.idx / 144;
        float xs = obj.idx % 144;
        float offset_x = ((float)offset_data_u8[obj.idx * 2 + 0] -
                          (float)offset_zero_point) *
                         offset_scale;
        float offset_y = ((float)offset_data_u8[obj.idx * 2 + 1] -
                          (float)offset_zero_point) *
                         offset_scale;
        xs += offset_x;
        ys += offset_y;

        float size_0 =
            ((float)size_data_u8[obj.idx * 4 + 0] - (float)size_zero_point) *
            size_scale;
        float size_1 =
            ((float)size_data_u8[obj.idx * 4 + 1] - (float)size_zero_point) *
            size_scale;
        float size_2 =
            ((float)size_data_u8[obj.idx * 4 + 2] - (float)size_zero_point) *
            size_scale;
        float size_3 =
            ((float)size_data_u8[obj.idx * 4 + 3] - (float)size_zero_point) *
            size_scale;
        float x1 = xs - size_0;
        float y1 = ys - size_1;
        float x2 = xs + size_2;
        float y2 = ys + size_3;
        x1 = (x1 * 4 - tmp_w) * ratio_x;
        y1 = (y1 * 4 - tmp_h) * ratio_y;
        x2 = (x2 * 4 - tmp_w) * ratio_x;
        y2 = (y2 * 4 - tmp_h) * ratio_y;

        x1 = std::max(std::min(x1, (float)(frame_width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(frame_height - 1)), 0.f);
        x2 = std::max(std::min(x2, (float)(frame_width - 1)), 0.f);
        y2 = std::max(std::min(y2, (float)(frame_height - 1)), 0.f);

        obj.rect.x = x1;
        obj.rect.y = y1;
        obj.rect.width = x2 - x1;
        obj.rect.height = y2 - y1;
        for (size_t i = 0; i < 128; i++) {
            // obj.feat(0, i) = ((float)feat_data_u8[obj.idx * 128 + i] -
            // (float)feat_zero_point) * feat_scale;
            obj.feat.at<float>(0, i) = ((float)feat_data_u8[obj.idx * 128 + i] -
                                        (float)feat_zero_point) *
                                       feat_scale;
        }
        cv::normalize(obj.feat, obj.feat, cv::NORM_L2);
    }
    nms_sorted_bboxes(objects, objects_pick, picked, 0.25);
    int count = picked.size();
    // fprintf(stderr, "pick_size: %d obj_size: %ld\n", count,objects.size());
    objects_pick.resize(count);

    std::vector<Track> tracks;
    JDETracker::instance()->update(objects_pick, &tracks);
    if (tracks.size() == 0) {
        MOTTrack mot_track;
        Rect ret = {0, 0, 0, 0};

        mot_track.ids = 1;
        mot_track.score = 0;
        mot_track.rects = ret;
        // result.push_back(mot_track);
    } else {
        std::vector<Track>::iterator titer;
        for (titer = tracks.begin(); titer != tracks.end(); ++titer) {
            if (titer->score < fairmotgraph->prob_threshold) {
                continue;
            } else {
                float w = titer->ltrb[2] - titer->ltrb[0];
                float h = titer->ltrb[3] - titer->ltrb[1];
                bool vertical = w / h > 1.6;
                float area = w * h;
                if (area > 200 && !vertical) {
                    MOTTrack mot_track;
                    Rect ret = {titer->ltrb[0], titer->ltrb[1], titer->ltrb[2],
                                titer->ltrb[3]};
                    mot_track.rects = ret;
                    mot_track.score = titer->score;
                    mot_track.ids = titer->id;
                    fairmotgraph->result.push_back(mot_track);
                }
            }
        }
    }
    return 0;
}
std::mutex gMutexVideo;
std::condition_variable gConditionVideo;
bool gRet = true;
void decodeVideo(struct FfmpegDemuxer* demuxer, struct Vim3Decoder* decoder,
                 struct VideoData* data) {
    while (gRet) {
        while (getFfmpegDemuxer(demuxer) == 0) {
            // double start_all = get_current_time();
            decoder->in_ptr = demuxer->packet->data;
            decoder->Readlen = demuxer->packet->size;

            if (sendVim3Decoder(decoder) < 0) {
                av_packet_unref(demuxer->packet);
                printf("write to decoder error\n");
                break;
            } else {
                do {
                    if (decoder->size ==
                        (demuxer->width * demuxer->height * 3 / 2)) {
                        // printf("line %d size %d\n", __LINE__, decoder->size);
                        cv::Mat yuvNV21;
                        yuvNV21.create(demuxer->height * 3 / 2, demuxer->width,
                                       CV_8UC1);
                        // printf("line %d\n", __LINE__);
                        memcpy(yuvNV21.data, decoder->out_ptr, decoder->size);
                        // printf("line %d\n", __LINE__);
                        std::unique_lock<std::mutex> lockerVideo(
                            gMutexVideo);  // 锁
                        while (!data->frame.empty())
                            gConditionVideo.wait(lockerVideo);
                        cvtColor(yuvNV21, data->frame, cv::COLOR_YUV2RGB_NV21);
                        data->size = decoder->size;
                        gConditionVideo.notify_all();  // 通知取
                    }
                } while (readVim3Decoder(decoder) > 0x100);
            }
        }
        gRet = false;
        break;
    }
}
std::mutex gMutexGrapgh;
bool gVideoFlag = false;
std::condition_variable gConditionGrapgh;

void runGraph(struct FairmotGraph* fairmotgraph, int frame_height,
              int frame_width, int time, struct VideoData* data,
              int frame_num) {
    while (gRet) {
        cv::Mat frame;

        {
            std::unique_lock<std::mutex> lockerVideo(gMutexVideo);  // 锁
            while (data->frame.empty()) gConditionVideo.wait(lockerVideo);
            frame = data->frame.clone();
            data->frame.release();
            gConditionVideo.notify_all();  // 通知取
        }

        const float mean[3] = {0, 0, 0};
        const float scale[3] = {0.003921, 0.003921, 0.003921};

        fprintf(stderr, "%d/%d\n", time++, frame_num);
        if (frame.empty()) {
            fprintf(stderr, "run img empty\n");
            gRet = false;
            return;
            break;
        }
        /* prepare process input data, set the data mem to input tensor */
        get_input_data_uint8(frame, fairmotgraph->input_data.data(), img_h,
                             img_w, mean, scale, fairmotgraph->input_scale,
                             fairmotgraph->input_zero_point);
        double start = get_current_time();
        /* run graph */
        {
            std::unique_lock<std::mutex> lockerGraph(gMutexGrapgh);  // 锁
            while (!data->outFrame.empty()) gConditionGrapgh.wait(lockerGraph);
            if (run_graph(fairmotgraph->graph, 1) < 0) {
                fprintf(stderr, "Run graph failed\n");
                break;
                // return -1;
            }
            data->outFrame = frame.clone();
            gVideoFlag = true;
            gConditionGrapgh.notify_all();  // 通知取
        }
        double end = get_current_time();
        double cur = end - start;
        fprintf(stdout, "Run graph cost %.2fms.\n", cur);
    }
}
void encodeVideo(struct FairmotGraph* fairmotgraph, int frame_height,
                 int frame_width, struct VideoData* data, int frame_num,
                 struct Vim3Encoder* encoder) {
    // std::string video_path = "./out/fairmot_timvx_out(0).mp4";
    // std::string video_name = "./out/fairmot_timvx_out";
    // cv::VideoWriter video_out("./out/fairmot_timvx_out(0).mp4", 828601953,
    // 20,
    //                           cv::Size(frame_width, frame_height), true);
    // int NUM = 0;
    // int time;
    while (gRet) {
        cv::Mat frame;

        {
            std::unique_lock<std::mutex> lockerGraph(gMutexGrapgh);  // 锁
            while (data->outFrame.empty()) gConditionGrapgh.wait(lockerGraph);
            frame = data->outFrame.clone();

            if (outFairmotGraph(fairmotgraph, frame_height, frame_width) < 0) {
                fprintf(stderr, "run graph failed \n");
                // return -1;
                break;
            }
            data->outFrame.release();
            gConditionGrapgh.notify_all();  // 通知取
        }
        cv::Mat vis_img = VisualizeTrackResult(frame, fairmotgraph->result);
        fairmotgraph->result.clear();
        double encode_start = get_current_time();
        if (!vis_img.empty()) {
            if ((getVim3Encoder(encoder, vis_img.data, frame_height,
                                frame_width)) < 0) {
                fprintf(stderr, "encoder failed \n");
                break;
            }
            // NUM = time/100;
            //             time++;
            // #ifdef CUT
            //             if (!(time % 400) || time == 1) {
            //                 std::cout << video_path << "\n";
            //                 video_path = video_name + "(" +
            //                 std::__cxx11::to_string(NUM++) +
            //                              ")" + ".mp4";
            //                 cv::VideoWriter video_out1(video_path, 828601953,
            //                 30,
            //                                            cv::Size(vis_img.cols,
            //                                            vis_img.rows), true);
            //                 video_out = video_out1;
            //                 video_out1.release();
            //             }
            // #endif

            //             if (!video_out.isOpened())
            //                 fprintf(stderr, "fairmot_timvx_out.mp4 can't
            //                 create");
            //             video_out.write(vis_img);

        } else {
            releaseVim3Encoder(encoder);
            // video_out.release();
            // goto close;
            // return -1;
            break;
        }
    }
}

int inputVideo(const char* video_file, const char* model_file,
               float prob_threshold) {
    std::vector<std::string> imagePathList;
    cv::VideoCapture cap;
    // cv::Mat frame;

    FfmpegDemuxer demuxer;
    Vim3Decoder decoder;
    Vim3Encoder encoder;
    int frame_width;
    int frame_height;
    int frame_num;
    double fps;

    /*decoder init */
    if (initFfmpegDemuxer(&demuxer) < 0) {
        printf("failed to init FfmpegDemuxer");
        releaseFfmpegDemuxer(&demuxer);
        printf("failed in line %d\n", __LINE__);
        return -1;
    }
    if (infoFfmpegDemuxer(video_file, &demuxer) < 0) {
        printf("failed to demuxing video");
        releaseFfmpegDemuxer(&demuxer);
        printf("failed in line %d\n", __LINE__);
        return -1;
    }
    fps = decoder.fps = encoder.framerate = demuxer.fps;
    frame_height = decoder.height = encoder.height = demuxer.height;
    frame_width = decoder.width = encoder.width = demuxer.width;
    frame_num = demuxer.nb_frames;
    encoder.gop = 10;

    if (initVim3Decoder(&decoder) < 0) {
        printf("failed to init decoder");
        printf("failed in line %d\n", __LINE__);
        return -1;
    }
    if (initVim3Encoder(&encoder) < 0) {
        fprintf(stderr, "faied to init encoder");
        return -1;
    }

    /*graph init*/
    FairmotGraph fairmotgraph;
    VideoData data;

    fairmotgraph.prob_threshold = prob_threshold;
    if ((initFairmotGraph(&fairmotgraph, model_file)) < 0) {
        fprintf(stderr, "graph init failed.\n");
        return -1;
    }
    fprintf(stdout, "video w: %d  h: %d  fps: %.2f\r\n", frame_width,
            frame_height, fps);
    int time = 0;

    std::thread DecodeThread(decodeVideo, &demuxer, &decoder, &data);
    std::thread GraphThread(runGraph, &fairmotgraph, frame_height, frame_width,
                            time, &data, frame_num);
    std::thread EncodeThread(encodeVideo, &fairmotgraph, frame_height,
                             frame_width, &data, frame_num, &encoder);
    DecodeThread.join();
    printf("DecodeThread over\n");
    releaseFfmpegDemuxer(&demuxer);
    releaseVim3Decoder(&decoder);
    GraphThread.join();
    printf("GraphThread over\n");
    EncodeThread.join();
    printf("EncodeThread over\n");

    // close:
    /* release tengine */
    releaseVim3Encoder(&encoder);

    postrun_graph(fairmotgraph.graph);
    destroy_graph(fairmotgraph.graph);
    release_tengine();

    return 0;
}
int main(int argc, char* argv[]) {
    const char* model_file = nullptr;
    const char* image_file = nullptr;
    const char* video_file = nullptr;

    int camera_num = 0;
    float find_thresh = 0.5;
    // bool state =  true ;
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
                    break;
                default:
                    break;
            }
        }
        fprintf(stderr, "thresh= %f\n", find_thresh);
        if (inputVideo(video_file, model_file, find_thresh) < 0) return -1;
    } else {
        fprintf(stderr, "Error: video or image file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file) || !check_file_exist(video_file))
        return -1;
}
