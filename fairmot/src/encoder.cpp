#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "vpcodec_1_0.h"
#ifdef __ANDROID__
#include <media/stagefright/foundation/ALooper.h>
using namespace android;

int main(int argc, const char *argv[]) {
    int width, height, gop, framerate, bitrate, num, in_size = 0;
    int outfd = -1;
    FILE *fp = NULL;
    int datalen = 0;
    int fmt = 0;
    vl_codec_handle_t handle_enc;
    int64_t total_encode_time = 0, t1, t2;
    int num_actually_encoded = 0;
    if (argc < 9) {
        printf("Amlogic AVC Encode API \n");
        printf(
            " usage: output "
            "[srcfile][outfile][width][height][gop][framerate][bitrate][num]["
            "fmt]\n");
        printf("  options  :\n");
        printf("  srcfile  : yuv data url in your root fs\n");
        printf("  outfile  : stream url in your root fs\n");
        printf("  width    : width\n");
        printf("  height   : height\n");
        printf("  gop      : I frame refresh interval\n");
        printf("  framerate: framerate \n ");
        printf("  bitrate  : bit rate \n ");
        printf("  num      : encode frame count \n ");
        printf(
            "  fmt      : encode input fmt 0:nv12 1:nv21 2:i420 (yu12) "
            "3:rgb888 4:bgr888\n ");
        return -1;
    } else {
        printf("%s\n", argv[1]);
        printf("%s\n", argv[2]);
    }
    width = atoi(argv[3]);
    if ((width < 1) || (width > 1920)) {
        printf("invalid width \n");
        return -1;
    }
    height = atoi(argv[4]);
    if ((height < 1) || (height > 1080)) {
        printf("invalid height \n");
        return -1;
    }
    gop = atoi(argv[5]);
    framerate = atoi(argv[6]);
    bitrate = atoi(argv[7]);
    num = atoi(argv[8]);
    fmt = atoi(argv[9]);
    if ((framerate < 0) || (framerate > 30)) {
        printf("invalid framerate %d \n", framerate);
        return -1;
    }
    if (bitrate <= 0) {
        printf("invalid bitrate \n");
        return -1;
    }
    if (num < 0) {
        printf("invalid num \n");
        return -1;
    }
    printf("src_url is: %s ;\n", argv[1]);
    printf("out_url is: %s ;\n", argv[2]);
    printf("width   is: %d ;\n", width);
    printf("height  is: %d ;\n", height);
    printf("gop     is: %d ;\n", gop);
    printf("frmrate is: %d ;\n", framerate);
    printf("bitrate is: %d ;\n", bitrate);
    printf("frm_num is: %d ;\n", num);

    unsigned framesize = width * height * 3 / 2;
    if (fmt == 4 || fmt == 3) {
        framesize = width * height * 3;
    }
    unsigned output_size = 1024 * 1024 * sizeof(char);
    unsigned char *output_buffer = (unsigned char *)malloc(output_size);
    unsigned char *input_buffer = (unsigned char *)malloc(framesize);

    fp = fopen((char *)argv[1], "rb");
    if (fp == NULL) {
        printf("open src file error!\n");
        goto exit;
    }
    outfd = open((char *)argv[2], O_CREAT | O_RDWR | O_TRUNC, 0644);
    if (outfd < 0) {
        printf("open dist file error!\n");
        goto exit;
    }
    handle_enc = vl_video_encoder_init(CODEC_ID_H264, width, height, framerate,
                                       bitrate, gop, IMG_FMT_YV12);
    while (num > 0) {
        if (fread(input_buffer, 1, framesize, fp) != framesize) {
            printf("read input file error!\n");
            goto exit;
        }
        memset(output_buffer, 0, output_size);
#ifdef __ANDROID__
        t1 = ALooper::GetNowUs();
#endif

        datalen =
            vl_video_encoder_encode(handle_enc, FRAME_TYPE_AUTO, input_buffer,
                                    in_size, output_buffer, fmt);

#ifdef __ANDROID__
        t2 = ALooper::GetNowUs();
        total_encode_time += t2 - t1;
#endif

        if (datalen >= 0) {
            num_actually_encoded++;
            write(outfd, (unsigned char *)output_buffer, datalen);
        }
        num--;
    }
    vl_video_encoder_destory(handle_enc);
    close(outfd);
    fclose(fp);
    free(output_buffer);
    free(input_buffer);

#ifdef __ANDROID__
    printf("total_encode_time: %lld, num_actually_encoded: %d, fps=%3.3f\n",
           total_encode_time, num_actually_encoded,
           num_actually_encoded * 1.0 * 1000000 / total_encode_time);
#endif

    return 0;
exit:
    if (input_buffer) free(input_buffer);
    if (output_buffer) free(output_buffer);
    if (outfd >= 0) close(outfd);
    if (fp) fclose(fp);
    return -1;
}
#endif
struct Vim3Encoder {
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
    FILE *outfd;
};
static double get_current_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
/// @brief 初始化编码器函数 定义输入视频数据的长宽与视频帧数等信息
/// @param encoder 存储了输入输出数据的信息
/// @return 若初始化成功返回0 失败返回-1
int initVim3Encoder(struct Vim3Encoder *encoder) {
    // FILE* outfd = NULL;
    long handle_enc;
    encoder->fmt = 0;
    encoder->gop = 10;
    encoder->framerate = 16;
    unsigned framesize = encoder->width * encoder->height * 3 / 2;
    if (encoder->fmt == 4 || encoder->fmt == 3) {
        framesize = encoder->width * encoder->height * 3;
    }
    unsigned output_size = 1024 * 1024 * sizeof(char);
    encoder->output_buffer = (unsigned char *)malloc(output_size);
    encoder->input_buffer = (unsigned char *)malloc(framesize);

    if ((encoder->outfd = fopen("./out/fairmot_timvx_out(0).h264", "wb")) ==
        NULL) {
        printf("./out/fairmot_timvx_out(0).h264 dump open file error!\n");
        return -1;
    }
    encoder->handle_enc = vl_video_encoder_init(
        CODEC_ID_H264, encoder->width, encoder->height, encoder->framerate,
        8000000, encoder->gop, IMG_FMT_YV12);
    if (encoder->handle_enc)
        return 0;
    else
        return -1;
}
/// @brief 将画框后的图像输入编码器
/// @param encoder 存储输入输出变量的结构
/// @param input 图像指针
/// @param height 图像高度
/// @param width 图像宽度
/// @return  返回送入编码器数据的长度
int getVim3Encoder(struct Vim3Encoder *encoder, const unsigned char *input,
                   int height, int width) {
    // memcpy(encoder->input_buffer,input ,width *height * 3);
    std::string video_name = "./out/fairmot_timvx_out";
    static int num = 0;
    double start = get_current_time();
    int datalen;
    datalen = vl_video_encoder_encode(
        encoder->handle_enc, FRAME_TYPE_AUTO, (unsigned char *)input,
        width * height * 3 / 2, encoder->output_buffer, 2);
    double end = get_current_time();
    fprintf(stdout, "encoder cost %.2fms.\n", end - start);
    fwrite(encoder->output_buffer, datalen, 1, encoder->outfd);
#ifdef CUT
    if (!(num++ % 200)) {
        fclose(encoder->outfd);
        std::string videoPath = video_name + "(" +
                                std::__cxx11::to_string(num / 400) + ")" +
                                ".h264";
        const char *path = (char *)videoPath.c_str();
        if ((encoder->outfd = fopen(path, "wb")) == NULL) {
            printf("./out/fairmot_timvx_out.h264  open  error!\n");
            return -1;
        }
    }
#endif

    return datalen;
}

void releaseVim3Encoder(struct Vim3Encoder *encoder) {
    vl_video_encoder_destory(encoder->handle_enc);
    fclose(encoder->outfd);
    free(encoder->output_buffer);
    free(encoder->input_buffer);
}