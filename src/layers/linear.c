#include <layers/linear.h>

#include <stdlib.h>

linear * linear_construct(const linear_params * params) {

	linear * obj = (linear *) malloc(sizeof(linear));

	obj->params = params;

	return obj;

}

void linear_destroy(linear * obj) {

	free((void *) obj);

}

int linear_forward(const linear * obj, const tensor * in, tensor * out) {

    const unsigned int num_dims_batch = in->num_dims1 * in->num_dims2 * in->num_dims3;
    const unsigned int num_dims_in = obj->params->num_dims_in;
    const unsigned int num_dims_out = obj->params->num_dims_out;

    for (unsigned int index_dim_batch = 0; index_dim_batch < num_dims_batch; index_dim_batch++) {

    	for (unsigned int index_dim_out = 0; index_dim_out < num_dims_out; index_dim_out++) {

    		float b = obj->params->b[index_dim_out];
    		float y = b;

    		for (unsigned int index_dim_in = 0; index_dim_in < num_dims_in; index_dim_in++) {

    			float w = obj->params->W[index_dim_out * num_dims_in + index_dim_in];
    			float x = in->data[index_dim_batch * num_dims_in + index_dim_in];
    			y += w * x;

    		}

    		out->data[index_dim_batch * num_dims_out + index_dim_out] = y;

    	}

	}	

	return 0;

}