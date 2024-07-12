#include <layers/sigmoid.h>

#include <math.h>
#include <stdlib.h>

sigmoid * sigmoid_construct(void) {

    sigmoid * obj = (sigmoid *) malloc(sizeof(sigmoid));

    return obj;

}

void sigmoid_destroy(sigmoid * obj) {

    free((void *) obj);

}

int sigmoid_forward(const sigmoid * obj, const tensor * in, tensor * out) {

    const unsigned int num_dims = in->num_dims1 * in->num_dims2 * in->num_dims3 * in->num_dims4;

    for (unsigned int index_dim = 0; index_dim < num_dims; index_dim++) {
        out->data[index_dim] = 1.0f / (1.0f + expf(-1.0f * in->data[index_dim]));
    }        

    return 0;

}