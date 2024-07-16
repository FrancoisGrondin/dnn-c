#include <layers/gru.h>

#include <stdlib.h>

ugru * ugru_construct(const ugru_params * params) {

	ugru * obj = (ugru *) malloc(sizeof(ugru));

	obj->params = params;

	obj->n = (float *) calloc(sizeof(float), params->num_dims_out);
	obj->r = (float *) calloc(sizeof(float), params->num_dims_out);
	obj->z = (float *) calloc(sizeof(float), params->num_dims_out);

	return obj;

}

void ugru_destroy(ugru * obj) {

	free((void *) obj->n);
	free((void *) obj->r);
	free((void *) obj->z);

	free((void *) obj);

}

int ugru_forward(const ugru * obj, const tensor * in, const tensor * hidden, tensor * out) {

	return 0;

}
