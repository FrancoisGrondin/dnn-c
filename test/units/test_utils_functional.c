#include "test_utils_functional.h"

int test_utils_functional_relu(void) {

    const float eps = 1e-5f;

    {

        const unsigned num_dims1 = 2;
        const unsigned num_dims2 = 3;
        const unsigned num_dims3 = 2;
        const unsigned num_dims4 = 4;

        tensor * in = tensor_construct(num_dims1, num_dims2, num_dims3, num_dims4);
        tensor * target = tensor_construct(num_dims1, num_dims2, num_dims3, num_dims4);
        tensor * pred = tensor_construct(num_dims1, num_dims2, num_dims3, num_dims4);
        
        const float in_array[] = { +1.0f, +2.0f, -3.0f, -2.0f, 
                                   +2.0f, +1.0f, +3.0f, +2.0f, 
                                   -3.0f, -4.0f, +0.0f, +1.0f, 
                                   -2.0f, -3.0f, -3.0f, +5.0f, 
                                   -1.0f, +3.0f, +2.0f, -1.0f,
                                   +1.0f, +1.0f, -3.0f, -3.0f,
                                   -2.0f, +1.0f, +0.0f, -2.0f,
                                   +4.0f, +1.0f, -2.0f, +0.0f,
                                   +0.0f, +3.0f, -1.0f, -1.0f,
                                   +2.0f, -2.0f, -1.0f, +0.0f,
                                   +0.0f, +2.0f, +1.0f, -4.0f, 
                                   +1.0f, +0.0f, -3.0f, +4.0f };

        const float target_array[] = { +1.0f, +2.0f, +0.0f, +0.0f, 
                                       +2.0f, +1.0f, +3.0f, +2.0f,
                                       +0.0f, +0.0f, +0.0f, +1.0f,
                                       +0.0f, +0.0f, +0.0f, +5.0f,
                                       +0.0f, +3.0f, +2.0f, +0.0f,
                                       +1.0f, +1.0f, +0.0f, +0.0f,
                                       +0.0f, +1.0f, +0.0f, +0.0f,
                                       +4.0f, +1.0f, +0.0f, +0.0f,
                                       +0.0f, +3.0f, +0.0f, +0.0f,
                                       +2.0f, +0.0f, +0.0f, +0.0f,
                                       +0.0f, +2.0f, +1.0f, +0.0f,
                                       +1.0f, +0.0f, +0.0f, +4.0f };

        tensor_load(in, in_array);
        tensor_load(target, target_array);
        
        relu(in, pred);

        if (tensors_compare(pred, target, eps) != 0) {
            return -1;
        }

        tensor_destroy(in);
        tensor_destroy(target);
        tensor_destroy(pred);

    }

    return 0;

}

int test_utils_functional_sigmoid(void) {

    const float eps = 1e-5f;

    {

        const unsigned num_dims1 = 1;
        const unsigned num_dims2 = 1;
        const unsigned num_dims3 = 2;
        const unsigned num_dims4 = 4;

        tensor * in = tensor_construct(num_dims1, num_dims2, num_dims3, num_dims4);
        tensor * target = tensor_construct(num_dims1, num_dims2, num_dims3, num_dims4);
        tensor * pred = tensor_construct(num_dims1, num_dims2, num_dims3, num_dims4);

        const float in_array[] = { +1.0f, -2.0f, +0.0f, +3.0f, 
                                   -100.0f, +100.0f, +10.0f, -10.0f };

        const float target_array[] = { +0.731059f, +0.119203f, +0.500000f, +0.952574f, 
                                       +0.000000f, +1.000000f, +0.999955f, +0.000045f };

        tensor_load(in, in_array);
        tensor_load(target, target_array);
        
        sigmoid(in, pred);

        if (tensors_compare(pred, target, eps) != 0) {
            return -1;
        }

        tensor_destroy(in);
        tensor_destroy(target);
        tensor_destroy(pred);

    }

    return 0;

}

int test_utils_functional_softmax(void) {

    return 0;

}