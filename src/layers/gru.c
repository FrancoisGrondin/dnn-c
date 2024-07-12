#include <layers/gru.h>

#include <stdlib.h>

ugru * ugru_construct(const unsigned int num_dims_in, const unsigned int num_dims_out) {

	ugru * obj = (ugru *) malloc(sizeof(ugru));

	obj->num_dims_in = num_dims_in;
	obj->num_dims_out = num_dims_out;

    obj->W_ir = (float *) calloc(sizeof(float), num_dims_in * num_dims_out);
    obj->b_ir = (float *) calloc(sizeof(float), num_dims_out);
    obj->W_hr = (float *) calloc(sizeof(float), num_dims_in * num_dims_out);
    obj->b_hr = (float *) calloc(sizeof(float), num_dims_out);
    obj->W_iz = (float *) calloc(sizeof(float), num_dims_in * num_dims_out);
    obj->b_iz = (float *) calloc(sizeof(float), num_dims_out);
    obj->W_hz = (float *) calloc(sizeof(float), num_dims_in * num_dims_out);
    obj->b_hz = (float *) calloc(sizeof(float), num_dims_out);
    obj->W_in = (float *) calloc(sizeof(float), num_dims_in * num_dims_out);
    obj->b_in = (float *) calloc(sizeof(float), num_dims_out);
    obj->W_hn = (float *) calloc(sizeof(float), num_dims_in * num_dims_out);
    obj->b_hn = (float *) calloc(sizeof(float), num_dims_out);

    obj->r = (float *) calloc(sizeof(float), num_dims_out);
    obj->z = (float *) calloc(sizeof(float), num_dims_out);
    obj->n = (float *) calloc(sizeof(float), num_dims_out);

	return obj;

}

void ugru_destroy(ugru * obj) {

	free((void *) obj->W_ir);
	free((void *) obj->b_ir);
	free((void *) obj->W_hr);
	free((void *) obj->b_hr);
	free((void *) obj->W_iz);
	free((void *) obj->b_iz);
	free((void *) obj->W_hz);
	free((void *) obj->b_hz);
	free((void *) obj->W_in);
	free((void *) obj->b_in);
	free((void *) obj->W_hn);
	free((void *) obj->b_hn);

	free((void *) obj->r);
	free((void *) obj->z);
	free((void *) obj->n);

	free((void *) obj);

}

int ugru_forward(const ugru * obj, const tensor * in, const tensor * hidden, tensor * out) {

	return 0;

}
