#include <layers/relu.h>

#include <stdlib.h>

relu * relu_construct(void) {

    relu * obj = (relu *) malloc(sizeof(relu));

    return obj;

}

void relu_destroy(relu * obj) {

    free((void *) obj);

}

int relu_forward(const relu * obj, const tensor * in, tensor * out) {

    const unsigned int num_dims = in->num_dims1 * in->num_dims2 * in->num_dims3 * in->num_dims4;

    for (unsigned int index_dim = 0; index_dim < num_dims; index_dim++) {
        out->data[index_dim] = (in->data[index_dim] > 0.0f) ? in->data[index_dim] : 0.0f;
    }

    return 0;

}