#ifndef __LAYERS_GRU
#define __LAYERS_GRU

#include "../utils/tensor.h"

typedef struct ugru_params {

    unsigned int num_dims_in;  // I
    unsigned int num_dims_out; // O

    float * W_ih; // 3O x I
    float * W_hh; // 3O x O
    float * b_ih; // 3O x 1
    float * b_hh; // 3O x 1

} ugru_params;

typedef struct ugru {

    const ugru_params * params;

    float * h;

} ugru;

ugru * ugru_construct(const ugru_params * params);

void ugru_destroy(ugru * obj);

int ugru_forward(const ugru * obj, const tensor * in, const tensor * hidden_in, tensor * hidden_out);

#endif // __LAYERS_GRU