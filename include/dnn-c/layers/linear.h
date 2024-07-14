#ifndef __LAYERS_LINEAR
#define __LAYERS_LINEAR

#include "../utils/tensor.h"

typedef struct linear_params {

    unsigned int num_dims_in;
    unsigned int num_dims_out;

    float * W;
    float * b;

} linear_params;

typedef struct linear {

    const linear_params * params;

} linear;

linear * linear_construct(const linear_params * params);

void linear_destroy(linear * obj);

int linear_forward(const linear * obj, const tensor * in, tensor * out);

#endif // __LAYERS_LINEAR