#include <layers/softmax.h>

#include <math.h>
#include <stdlib.h>

softmax * softmax_construct(void) {

    softmax * obj = (softmax *) malloc(sizeof(softmax));

    return obj;

}

void softmax_destroy(softmax * obj) {

    free((void *) obj);

}

int softmax_forward(const softmax * obj, const tensor * in, tensor * out) {

    const unsigned int num_dims_batch = in->num_dims1 * in->num_dims2 * in->num_dims3;
    const unsigned int num_dims_tensor = in->num_dims4;

    for (unsigned int index_dim_batch = 0; index_dim_batch < num_dims_batch; index_dim_batch++) {
        
        float max_value = -INFINITY;

        for (unsigned int index_dim_tensor = 0; index_dim_tensor < num_dims_tensor; index_dim_tensor++) {
            
            float x = in->data[index_dim_batch * num_dims_tensor + index_dim_tensor];
            max_value = (x > max_value) ? x : max_value;

        }

        float total = 0.0f;

        for (unsigned int index_dim_tensor = 0; index_dim_tensor < num_dims_tensor; index_dim_tensor++) {

            float x = in->data[index_dim_batch * num_dims_tensor + index_dim_tensor];
            float y = expf(x - max_value);
            total += y;

            out->data[index_dim_batch * num_dims_tensor + index_dim_tensor] = y;

        }

        for (unsigned int index_dim_tensor = 0; index_dim_tensor < num_dims_tensor; index_dim_tensor++) {

            out->data[index_dim_batch * num_dims_tensor + index_dim_tensor] /= total;

        }

    }        

    return 0;

}