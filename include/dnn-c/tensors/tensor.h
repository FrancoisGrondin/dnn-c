#ifndef __TENSORS_TENSOR
#define __TENSORS_TENSOR

    typedef struct tensor {

        unsigned int num_dims1;
        unsigned int num_dims2;
        unsigned int num_dims3;
        unsigned int num_dims4;
        float * data;

    } tensor;

    tensor * tensor_construct(unsigned int num_dims1, unsigned int num_dims2, unsigned int num_dims3, unsigned int num_dims4);

    void tensor_destroy(tensor * obj);

#endif // __TENSORS_TENSOR