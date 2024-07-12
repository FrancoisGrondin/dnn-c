#ifndef __LAYERS_SIGMOID
#define __LAYERS_SIGMOID

#include "../tensors/tensor.h"

typedef struct sigmoid {

    // Empty

} sigmoid;

sigmoid * sigmoid_construct(void);

void sigmoid_destroy(sigmoid * obj);

int sigmoid_forward(const sigmoid * obj, const tensor * in, tensor * out);

#endif // __LAYERS_SIGMOID