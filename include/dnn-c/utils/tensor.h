#ifndef __UTILS_TENSOR
#define __UTILS_TENSOR

typedef struct tensor {

    unsigned int num_dims1;
    unsigned int num_dims2;
    unsigned int num_dims3;
    unsigned int num_dims4;
    float * data;

} tensor;

tensor * tensor_construct(unsigned int num_dims1, unsigned int num_dims2, unsigned int num_dims3, unsigned int num_dims4);

void tensor_destroy(tensor * obj);

void tensor_load_1d(tensor * obj, const unsigned int index_dim1, const unsigned int index_dim2, const unsigned int index_dim3, const float * data);

void tensor_save_1d(const tensor * obj, const unsigned int index_dim1, const unsigned int index_dim2, const unsigned int index_dim3, float * data);

void tensor_printf(const tensor * obj);

#endif // __UTILS_TENSOR