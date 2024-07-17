#ifndef __LAYERS_GRU
#define __LAYERS_GRU

#include "../utils/tensor.h"

typedef struct ugru_params {

    unsigned int num_dims_in;  // I
    unsigned int num_dims_out; // O

    float * W_ir; // O x I
    float * b_ir; // O x 1
    float * W_hr; // O x O
    float * b_hr; // O x 1
    float * W_iz; // O x I
    float * b_iz; // O x 1
    float * W_hz; // O x O
    float * b_hz; // O x 1
    float * W_in; // O x I
    float * b_in; // O x 1
    float * W_hn; // O x O
    float * b_hn; // O x 1

} ugru_params;

typedef struct ugru {

    const ugru_params * params;

    float * h;

} ugru;

ugru * ugru_construct(const ugru_params * params);

void ugru_destroy(ugru * obj);

int ugru_forward(const ugru * obj, const tensor * in, const tensor * hidden, tensor * out);

#endif // __LAYERS_GRU