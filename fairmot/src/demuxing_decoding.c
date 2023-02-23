#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
#include <libavutil/timestamp.h>
#include <stdio.h>

#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
// #include "demuxing.h"

// MPEG-TS文件解封装得到的码流可直接播放
// MP4/FLV/MKV解封装得到的码流不可播放；
// 这与容器的封装方式有关。
/**
 * @brief  ffmpeg_demuxer对象存放解复用的结构体以及解复用后的packet
 */
struct FfmpegDemuxer {
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
int initFfmpegDemuxer(struct FfmpegDemuxer *demuxer) {
    demuxer->fmt_ctx = avformat_alloc_context();  // 包含h264数据的结构体
    if (demuxer->fmt_ctx == NULL) {
        printf("failed to alloc format context\n");
        return -1;
    }

    demuxer->packet = av_packet_alloc();  // 存放转换出的h264数据
    if (demuxer->packet == NULL) {
        printf("failed to alloc format context\n");
        return -1;
    }

    demuxer->bsf = av_bsf_get_by_name("h264_mp4toannexb");
    if (demuxer->bsf == NULL) {
        printf("failed to find stream filter\n");
        return -1;
    }
    av_bsf_alloc(demuxer->bsf, &(demuxer->bsf_ctx));
    return 0;
}
/// @brief 获取视频文件解复用的信息
/// @param url 视频文件的地址或网络连接
/// @param demuxer 解复用结构体
/// @return 失败返回-1 成功返回video_index
int infoFfmpegDemuxer(const char *url, struct FfmpegDemuxer *demuxer) {
    // 打开输入流
    if (avformat_open_input(&(demuxer->fmt_ctx), url, NULL, NULL) < 0) {
        printf("failed to open input url\n");
        return -1;
    }
    // 读取媒体文件信息
    if (avformat_find_stream_info(demuxer->fmt_ctx, NULL) < 0) {
        printf("failed to find stream\n");
        if (demuxer->fmt_ctx) avformat_close_input(&(demuxer->fmt_ctx));
        return -1;
    }
    av_dump_format(demuxer->fmt_ctx, 0, url, 0);

    // 寻找音频流和视频流下标
    demuxer->video_index = av_find_best_stream(
        demuxer->fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    printf("video index: %d\n", demuxer->video_index);
    if (demuxer->video_index < 0) {
        printf("failed to find stream index\n");
        if (demuxer->fmt_ctx) avformat_close_input(&(demuxer->fmt_ctx));
        return -1;
    }
    demuxer->nb_frames =
        demuxer->fmt_ctx->streams[demuxer->video_index]->nb_frames;
    demuxer->fps =
        (double)demuxer->fmt_ctx->streams[demuxer->video_index]
            ->avg_frame_rate.num /
        demuxer->fmt_ctx->streams[demuxer->video_index]->avg_frame_rate.den;
    demuxer->width =
        demuxer->fmt_ctx->streams[demuxer->video_index]->codecpar->width;
    demuxer->height =
        demuxer->fmt_ctx->streams[demuxer->video_index]->codecpar->height;
    // 初始化过滤器
    avcodec_parameters_copy(
        demuxer->bsf_ctx->par_in,
        demuxer->fmt_ctx->streams[demuxer->video_index]->codecpar);
    av_bsf_init(demuxer->bsf_ctx);
    return demuxer->video_index;
}

/// @brief 获取解复用后的packet
/// @param demuxer 视频数据存放在packet对象中
/// @return 获取成功返回0 失败返回-1
int getFfmpegDemuxer(struct FfmpegDemuxer *demuxer) {
    int errnum;
    char errnum_buffer[128];
    static int packet_num;
    if ((errnum = av_read_frame(demuxer->fmt_ctx, demuxer->packet)) == 0) {
        // printf("line %d stream_index %d
        // \n",__LINE__,demuxer->packet->stream_index);
        if (demuxer->packet->stream_index == demuxer->video_index) {
            // printf("line %d stream_index %d
            // \n",__LINE__,demuxer->video_index);
            if (av_bsf_send_packet(demuxer->bsf_ctx, demuxer->packet) == 0) {
                if (av_bsf_receive_packet(demuxer->bsf_ctx, demuxer->packet) ==
                    0) {
                    printf("num: %d the packet size: %d \n", packet_num++,
                           demuxer->packet->size);
                }
            }
        } else {
            av_packet_unref(demuxer->packet);
            // return -1;
        }
    } else {
        av_packet_unref(demuxer->packet);
        av_strerror(errnum, errnum_buffer, sizeof(errnum_buffer));
        printf(" error: %s\n", errnum_buffer);
        return -1;
    }

    return 0;
}

/// @brief 释放ffmpeg_demuxer结构体相关的资源
/// @param demuxer
/// @return 释放成功返回0 失败返回-1
int releaseFfmpegDemuxer(struct FfmpegDemuxer *demuxer) {
    if (demuxer->fmt_ctx) avformat_close_input(&(demuxer->fmt_ctx));
    if (demuxer->packet) av_packet_free(&(demuxer->packet));
    if (demuxer->bsf_ctx) av_bsf_free(&(demuxer->bsf_ctx));
    return 0;
}
