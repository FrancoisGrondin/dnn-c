#ifndef __LAYERS_GRU
#define __LAYERS_GRU

#include "../utils/tensor.h"

typedef struct ugru_params {

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

} ugru_params;

typedef struct ugru {

    const ugru_params * params;

    float * h;

} ugru;

ugru * ugru_construct(const ugru_params * params);

void ugru_destroy(ugru * obj);

int ugru_forward(const ugru * obj, const tensor * in, const tensor * hidden, tensor * out);

#endif // __LAYERS_GRU