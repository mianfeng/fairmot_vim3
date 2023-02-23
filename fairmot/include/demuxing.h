extern "C" {
#ifndef DEMUXING_H
#define DEMUXING_H
#endif

#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
#include <libavutil/timestamp.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <errno.h>
#include <stdbool.h>
#include <ctype.h>
#include <codec.h>
#include <amvideo.h>
#include <ion/IONmem.h>


/** 
 * @brief  ffmpeg_demuxer对象存放解复用的结构体以及解复用后的packet
 */ 
struct FfmpegDemuxer{
    /*解复用的视频包*/
    AVPacket *packet;
    /*解复用相关的结构体*/
    AVFormatContext *fmt_ctx;
    /*h264过滤器*/
   const AVBitStreamFilter *bsf;
    AVBSFContext *bsf_ctx;
    /*视频的编号信息*/
    int video_index;
    /*视频的帧数*/
    int64_t nb_frames;
    /*视频的帧率*/
    double fps;
    /*width*/
    int width;
    /*height*/
    int height;
};
/// @brief 将解封装用到的结构体进行初始化
/// @param demuxer ffmpeg_demuxer对象存放解复用的结构体以及解复用后的packet
/// @return 成功返回0
int initFfmpegDemuxer(struct FfmpegDemuxer *demuxer);

/// @brief 获取视频文件解复用的信息
/// @param url 视频文件的地址或网络连接
/// @param demuxer 解复用结构体
/// @return 失败返回-1 成功返回video_index
int infoFfmpegDemuxer(const char *url,struct FfmpegDemuxer *demuxer);

/// @brief 获取解复用后的packet 
/// @param demuxer 视频数据存放在packet对象中
/// @return 获取成功返回0 失败返回-1
int getFfmpegDemuxer(struct FfmpegDemuxer *demuxer);

/// @brief 释放ffmpeg_demuxer结构体相关的资源
/// @param demuxer 
/// @return 释放成功返回0 失败返回-1
int releaseFfmpegDemuxer(struct FfmpegDemuxer *demuxer);
}