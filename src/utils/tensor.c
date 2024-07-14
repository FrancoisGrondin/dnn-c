#include <utils/tensor.h>

#include <stdlib.h>

tensor * tensor_construct(unsigned int num_dims1, unsigned int num_dims2, unsigned int num_dims3, unsigned int num_dims4) {

	tensor * obj = (tensor *) malloc(sizeof(tensor));

	obj->num_dims1 = num_dims1;
	obj->num_dims2 = num_dims2;
	obj->num_dims3 = num_dims3;
	obj->num_dims4 = num_dims4;

	obj->data = (float *) calloc(sizeof(float), num_dims1 * num_dims2 * num_dims3 * num_dims4);

	return obj;

}

void tensor_destroy(tensor * obj) {

	free((void *) obj->data);

	free((void *) obj);

}