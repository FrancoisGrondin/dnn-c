#ifndef __LAYERS_RELU
#define __LAYERS_RELU

#include "../tensors/tensor.h"

typedef struct relu {

    // Empty

} relu;

relu * relu_construct(void);

void relu_destroy(relu * obj);

int relu_forward(const relu * obj, const tensor * in, tensor * out);

#endif // __LAYERS_RELU