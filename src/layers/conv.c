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

	return 0;

}