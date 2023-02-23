extern "C" {
#ifndef __IONPLAYER_H__
#define __IONPLAYER_H__
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
#define READ_SIZE (256 * 1024)
#define EXTERNAL_PTS    (1)
#define SYNC_OUTSIDE    (2)
#define UNIT_FREQ       96000
#define PTS_FREQ        90000
#define AV_SYNC_THRESH    PTS_FREQ*30
#define MESON_BUFFER_SIZE 4

extern struct out_buffer_t {
    int index;
    int size;
    bool own_by_v4l;
    void *ptr;
    IONMEM_AllocParams buffer;
} vbuffer[MESON_BUFFER_SIZE];

/// @brief 存放输入和输出解码器的数据及视频信息
struct Vim3Decoder {
    /// @brief 输出数据的指针
    void *out_ptr;
    /// @brief 输出数据的大小
    int size;
    /// @brief 输入数据的指针
    void *in_ptr;
    /// @brief 输入数据的大小
    uint32_t   Readlen;
    int   ret;
    int   width;
    int   height;
    int   fps;
    codec_para_t *pcodec;

};
/// @brief 对vim3_decoderd的初始话
/// @param decoder 对象包含解复用后数据，在ptr中
/// @return 成功返回0 失败返回-1
int initVim3Decoder(struct Vim3Decoder *decoder);

/// @brief 将数据送入解码器并返回解码后的yuv格式数据 
/// @param decoder 对象包含解码后数据，在out_buffer中
/// @return 失败返回-1 成功则返回写入解码器的数据
int sendVim3Decoder(struct Vim3Decoder *decoder);

/// @brief 从解码后的队列中取出数据，一次读取可能因关键帧没有获取数据，因此需要多次读取至数据读完
/// @param decoder 解码后的数据存放在ptr中
/// @return 返回剩余的数据长度
int readVim3Decoder(struct Vim3Decoder *decoder);

int releaseVim3Decoder(struct Vim3Decoder *decoder);

}