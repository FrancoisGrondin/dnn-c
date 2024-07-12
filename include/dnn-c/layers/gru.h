#ifndef __LAYERS_GRU
#define __LAYERS_GRU

#include "../tensors/tensor.h"

typedef struct ugru {

    unsigned int num_dims_in;
    unsigned int num_dims_out;

    float * W_ir;
    float * b_ir;
    float * W_hr;
    float * b_hr;
    float * W_iz;
    float * b_iz;
    float * W_hz;
    float * b_hz;
    float * W_in;
    float * b_in;
    float * W_hn;
    float * b_hn;

    float * r;
    float * z;
    float * n;

} ugru;

ugru * ugru_construct(const unsigned int num_dims_in, const unsigned int num_dims_out);

void ugru_destroy(ugru * obj);

int ugru_forward(const ugru * obj, const tensor * in, const tensor * hidden, tensor * out);

#endif // __LAYERS_GRU