#include "test_layers_gru.h"

int test_layers_gru(void) {

    const float eps = 1e-5f;

    {

        const unsigned in_num_dims1 = 1;
        const unsigned in_num_dims2 = 1;
        const unsigned in_num_dims3 = 1;
        const unsigned in_num_dims4 = 1;

        const unsigned out_num_dims1 = 1;
        const unsigned out_num_dims2 = 1;
        const unsigned out_num_dims3 = 1;
        const unsigned out_num_dims4 = 2;

        tensor * in = tensor_construct(in_num_dims1, in_num_dims2, in_num_dims3, in_num_dims4);
        tensor * hidden = tensor_construct(out_num_dims1, out_num_dims2, out_num_dims3, out_num_dims4);
        tensor * target = tensor_construct(out_num_dims1, out_num_dims2, out_num_dims3, out_num_dims4);
        tensor * pred = tensor_construct(out_num_dims1, out_num_dims2, out_num_dims3, out_num_dims4);

        const float in_array[] = { -4.0f };
        const float hidden_array[] = { +1.0f, +4.0f };
        const float target_array[] = { +0.7616f, -1.0000f };

        tensor_load(in, in_array);
        tensor_load(hidden, hidden_array);
        tensor_load(target, target_array);
        
        const ugru_params params = { .num_dims_in = 1,
                                     .num_dims_out = 2,
                                     .W_ih = (float []) {+5.00000000f,-3.00000000f,+2.00000000f,+1.00000000f,
                                                         +1.00000000f,+1.00000000f},
                                     .W_hh = (float []) {+1.00000000f,-2.00000000f,+3.00000000f,-4.00000000f,
                                                         +5.00000000f,-6.00000000f,+7.00000000f,-8.00000000f,
                                                         +9.00000000f,+0.00000000f,+1.00000000f,-2.00000000f},
                                     .b_ih = (float []) {+1.00000000f,+2.00000000f,+3.00000000f,+4.00000000f,
                                                         +5.00000000f,+6.00000000f},
                                     .b_hh = (float []) {+3.00000000f,+6.00000000f,+9.00000000f,-2.00000000f,
                                                         -5.00000000f,-8.00000000f} };

        ugru * layer = ugru_construct(&params);



        ugru_forward(layer, in, hidden, pred);

        if (tensors_compare(pred, target, eps) != 0) {
            return -1;
        }

        tensor_destroy(in);
        tensor_destroy(hidden);
        tensor_destroy(target);
        tensor_destroy(pred);

    }

    return 0;

}