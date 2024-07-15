#include "test_layers_linear.h"

int test_layers_linear(void) {

    const float eps = 1e-5f;

    {

        const unsigned in_num_dims1 = 1;
        const unsigned in_num_dims2 = 1;
        const unsigned in_num_dims3 = 2;
        const unsigned in_num_dims4 = 4;

        const unsigned out_num_dims1 = 1;
        const unsigned out_num_dims2 = 1;
        const unsigned out_num_dims3 = 2;
        const unsigned out_num_dims4 = 3;

        tensor * in = tensor_construct(in_num_dims1, in_num_dims2, in_num_dims3, in_num_dims4);
        tensor * target = tensor_construct(out_num_dims1, out_num_dims2, out_num_dims3, out_num_dims4);
        tensor * pred = tensor_construct(out_num_dims1, out_num_dims2, out_num_dims3, out_num_dims4);

        const float in_array[] = { +1.0f, -2.0f, +0.0f, +3.0f, 
                                   -100.0f, +100.0f, +10.0f, -10.0f };

        const float target_array[] = { +11.0f, +16.0f, +6.0f, 
                                       +92.0f, +89.0f, -986.0f };

        tensor_load(in, in_array);
        tensor_load(target, target_array);
        
        const linear_params params = { .num_dims_in = in_num_dims4, 
                                       .num_dims_out = out_num_dims4,
                                       .W = (float []) { +1.0f, +2.0f, +3.0f, +4.0f, 
                                                         +5.0f, +6.0f, +7.0f, +8.0f, 
                                                         +9.0f, -1.0f, -2.0f, -3.0f },
                                       .b = (float []) { +2.0f, -1.0f, +4.0f } };

        linear * layer = linear_construct(&params);

        linear_forward(layer, in, pred);

        if (tensors_compare(pred, target, eps) != 0) {
            return -1;
        }

        tensor_destroy(in);
        tensor_destroy(target);
        tensor_destroy(pred);

    }

    return 0;

}