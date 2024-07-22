#include <layers/conv.h>

#include <stdlib.h>

conv2d * conv2d_construct(const conv2d_params * params) {

	conv2d * obj = (conv2d *) malloc(sizeof(conv2d));

	obj->params = params;

	return obj;

}

void conv2d_destroy(conv2d * obj) {

	free((void *) obj);

}

int conv2d_forward(const conv2d * obj, const tensor * in, tensor * out) {

    const unsigned int num_dims_batch = in->num_dims1;
    const unsigned int num_dims_channels_in = obj->channels_in;
    const unsigned int num_dims_channels_out = obj->channels_out;
    const unsigned int num_dims_height_in = in->num_dims3;
    const unsigned int num_dims_width_in = in->num_dims4;
    const unsigned int num_dims_height_out = out->num_dims3;
    const unsigned int num_dims_width_out = out->num_dims4;

	for (unsigned int index_dim_batch = 0; index_dim_batch < num_dims_batch; index_dim_batch++) {
		
	}

	return 0;

}