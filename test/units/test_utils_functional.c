#include "test_utils_functional.h"

int test_utils_functional_relu(void) {

    {

        const float eps = 1e-5f;

        const unsigned num_dims1 = 2;
        const unsigned num_dims2 = 3;
        const unsigned num_dims3 = 2;
        const unsigned num_dims4 = 4;

        tensor * in;
        tensor * out;

        in = tensor_construct(num_dims1, num_dims2, num_dims3, num_dims4);
        out = tensor_construct(num_dims1, num_dims2, num_dims3, num_dims4);
        
        const float in_array[2][3][2][4] =
            { { { { +1.0f, +2.0f, -3.0f, -2.0f }, { +2.0f, +1.0f, +3.0f, +2.0f } }, { { -3.0f, -4.0f, +0.0f, +1.0f }, { -2.0f, -3.0f, -3.0f, +5.0f } }, { { -1.0f, +3.0f, +2.0f, -1.0f }, { +1.0f, +1.0f, -3.0f, -3.0f } } },
              { { { -2.0f, +1.0f, +0.0f, -2.0f }, { +4.0f, +1.0f, -2.0f, +0.0f } }, { { +0.0f, +3.0f, -1.0f, -1.0f }, { +2.0f, -2.0f, -1.0f, +0.0f } }, { { +0.0f, +2.0f, +1.0f, -4.0f }, { +1.0f, +0.0f, -3.0f, +4.0f } } } };

        const float out_array[2][3][2][4] =
            { { { { +1.0f, +2.0f, +0.0f, +0.0f }, { +2.0f, +1.0f, +3.0f, +2.0f } }, { { +0.0f, +0.0f, +0.0f, +1.0f }, { +0.0f, +0.0f, +0.0f, +5.0f } }, { { +0.0f, +3.0f, +2.0f, +0.0f }, { +1.0f, +1.0f, +0.0f, +0.0f } } },
              { { { +0.0f, +1.0f, +0.0f, +0.0f }, { +4.0f, +1.0f, +0.0f, +0.0f } }, { { +0.0f, +3.0f, +0.0f, +0.0f }, { +2.0f, +0.0f, +0.0f, +0.0f } }, { { +0.0f, +2.0f, +1.0f, +0.0f }, { +1.0f, +0.0f, +0.0f, +4.0f } } } };

        for (unsigned int index_dim1 = 0; index_dim1 < num_dims1; index_dim1++) {
            for (unsigned int index_dim2 = 0; index_dim2 < num_dims2; index_dim2++) {
                for (unsigned int index_dim3 = 0; index_dim3 < num_dims3; index_dim3++) {
                    tensor_load_1d(in, index_dim1, index_dim2, index_dim3, in_array[index_dim1][index_dim2][index_dim3]);
                }
            }
        }

        relu(in, out);

        for (unsigned int index_dim1 = 0; index_dim1 < num_dims1; index_dim1++) {
            for (unsigned int index_dim2 = 0; index_dim2 < num_dims2; index_dim2++) {
                for (unsigned int index_dim3 = 0; index_dim3 < num_dims3; index_dim3++) {
                    float pred[4] = { 0 };
                    tensor_save_1d(out, index_dim1, index_dim2, index_dim3, pred);
                    for (unsigned int index_dim4 = 0; index_dim4 < num_dims4; index_dim4++) {
                        if (fabsf(pred[index_dim4] - out_array[index_dim1][index_dim2][index_dim3][index_dim4]) > eps) {
                            return -1;
                        }
                    }
                }
            }
        }

    }

    return 0;

}