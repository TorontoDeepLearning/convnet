#ifndef COMMON_H
#define COMMON_H

typedef struct Shape4D
{
    int shape[4];
} Shape4D;

typedef struct ConvDesc
{
    int num_input_channels;
    int num_output_channels;
    int kernel_size_y;
    int kernel_size_x;
    int kernel_size_t;
    int stride_y;
    int stride_x;
    int stride_t;
    int padding_y;
    int padding_x;
    int padding_t;
    int input_channel_begin;
    int input_channel_end;
    int output_channel_begin;
    int output_channel_end;
    int num_groups;
} ConvDesc;

#endif

