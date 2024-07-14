#include <utils/tensor.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

void tensor_load_1d(tensor * obj, const unsigned int index_dim1, const unsigned int index_dim2, const unsigned int index_dim3, const float * data) {

	const unsigned int index_dim = ((index_dim1 * obj->num_dims2 + index_dim2) * obj->num_dims3 + index_dim3) * obj->num_dims4;

	memcpy(&(obj->data[index_dim]), data, sizeof(float) * obj->num_dims4);

}

void tensor_save_1d(const tensor * obj, const unsigned int index_dim1, const unsigned int index_dim2, const unsigned int index_dim3, float * data) {

	const unsigned int index_dim = ((index_dim1 * obj->num_dims2 + index_dim2) * obj->num_dims3 + index_dim3) * obj->num_dims4;

	memcpy(data, &(obj->data[index_dim]), sizeof(float) * obj->num_dims4);

}

void tensor_printf(const tensor * obj) {

	for (unsigned int index_dim1 = 0; index_dim1 < obj->num_dims1; index_dim1++) {
		for (unsigned int index_dim2 = 0; index_dim2 < obj->num_dims2; index_dim2++) {
			for (unsigned int index_dim3 = 0; index_dim3 < obj->num_dims3; index_dim3++) {
				for (unsigned int index_dim4 = 0; index_dim4 < obj->num_dims4; index_dim4++) {
					const unsigned int index_dim = ((index_dim1 * obj->num_dims2 + index_dim2) * obj->num_dims3 + index_dim3) * obj->num_dims4 + index_dim4;
					printf("(%u, %u, %u, %u): %+f\n", index_dim1, index_dim2, index_dim3, index_dim4, obj->data[index_dim]);
				}
			}
		}
	}

}