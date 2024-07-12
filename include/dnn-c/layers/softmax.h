#ifndef __LAYERS_SOFTMAX
#define __LAYERS_SOFTMAX

#include "../tensors/tensor.h"

typedef struct softmax {

    // Empty

} softmax;

softmax * softmax_construct(void);

void softmax_destroy(softmax * obj);

int softmax_forward(const softmax * obj, const tensor * in, tensor * out);

#endif // __LAYERS_SOFTMAX