#ifndef __UTILS_FUNCTIONAL
#define __UTILS_FUNCTIONAL

#include "tensor.h"

int relu(const tensor * in, tensor * out);

int sigmoid(const tensor * in, tensor * out);

int softmax(const tensor * in, tensor * out);

#endif // __UTILS_FUNCIONAL