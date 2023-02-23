#ifndef __ENCODER_H__
#define __ENCODER_H__
#endif
#include "vpcodec_1_0.h"

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>

struct Vim3Encoder
{
    /*输入视频的宽*/
    int width;
    /*输入视频的高*/
    int height;
    /*I帧的间隔*/
    int gop;
    /*视频的帧率*/
    int framerate;
    int in_size;
    unsigned char *output_buffer;
    unsigned char *input_buffer;   
    int fmt;
    long handle_enc;
    FILE* outfd;
};

int initVim3Encoder(struct Vim3Encoder *encoder);

int getVim3Encoder(struct Vim3Encoder *encoder,const unsigned char * input,int height,int width);

void releaseVim3Encoder(struct Vim3Encoder *encoder);

