#ifndef __LAYERS_CONV
#define __LAYERS_CONV

#include "../utils/tensor.h"

typedef struct conv2d_params {

    unsigned int channels_in;
    unsigned int channels_out;
    unsigned int kernel_size_x;
    unsigned int kernel_size_y;
    unsigned int padding_x;
    unsigned int padding_y;
    unsigned int stride_x;
    unsigned int stride_y;

    float * W;
    float * b;

} conv2d_params;

typedef struct conv2d {

    const conv2d_params * params;

} conv2d;

conv2d * conv2d_construct(const conv2d_params * params);

void conv2d_destroy(conv2d * obj);

int conv2d_forward(const conv2d * obj, const tensor * in, tensor * out);

#endif // __LAYERS_CONV