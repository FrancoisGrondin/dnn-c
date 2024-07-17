#ifndef __LAYERS_LINEAR
#define __LAYERS_LINEAR

#include "../utils/tensor.h"

typedef struct linear_params {

    unsigned int num_dims_in;  // I
    unsigned int num_dims_out; // O

    float * W; // O x I
    float * b; // O x 1

} linear_params;

typedef struct linear {

    const linear_params * params;

} linear;

linear * linear_construct(const linear_params * params);

void linear_destroy(linear * obj);

int linear_forward(const linear * obj, const tensor * in, tensor * out);

#endif // __LAYERS_LINEAR