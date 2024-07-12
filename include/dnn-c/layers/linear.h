#ifndef __LAYERS_LINEAR
#define __LAYERS_LINEAR

#include "../tensors/tensor.h"

typedef struct linear {

    unsigned int num_dims_in;
    unsigned int num_dims_out;

    float * W;
    float * b;

} linear;

linear * linear_construct(const unsigned int num_dims_in, const unsigned int num_dims_out);

void linear_destroy(linear * obj);

int linear_forward(const linear * obj, const tensor * in, tensor * out);

#endif // __LAYERS_LINEAR