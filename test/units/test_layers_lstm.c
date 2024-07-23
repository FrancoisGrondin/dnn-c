#include "test_layers_lstm.h"

int test_layers_lstm(void) {

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

        tensor * x_in = tensor_construct(in_num_dims1, in_num_dims2, in_num_dims3, in_num_dims4);
        tensor * hidden_in = tensor_construct(out_num_dims1, out_num_dims2, out_num_dims3, out_num_dims4);
        tensor * cell_in = tensor_construct(out_num_dims1, out_num_dims2, out_num_dims3, out_num_dims4);
        tensor * hidden_target = tensor_construct(out_num_dims1, out_num_dims2, out_num_dims3, out_num_dims4);
        tensor * cell_target = tensor_construct(out_num_dims1, out_num_dims2, out_num_dims3, out_num_dims4);
        tensor * hidden_pred = tensor_construct(out_num_dims1, out_num_dims2, out_num_dims3, out_num_dims4);
        tensor * cell_pred = tensor_construct(out_num_dims1, out_num_dims2, out_num_dims3, out_num_dims4);

        const float x_in_array[] = {+0.66135216f,+0.26692411f,+0.06167726f,+0.62131733f,
                                    -0.45190597f,-0.16613023f,-1.52276850f,+0.38168392f};
        const float hidden_in_array[] = {-1.02760863f,-0.56305277f,-0.89229053f,
                                         -0.05825018f,-0.19550958f,-0.96563596f};
        const float cell_in_array[] = {+0.42241532f,+0.26731700f,-0.42119515f,
                                       -0.51069999f,-1.57266521f,-0.12324776f};
        const float hidden_target_array[] = {-0.16337124f,-0.17244922f,-0.11701180f,
                                             -0.24484144f,-0.48873585f,+0.03004458f};
        const float cell_target_array[] = {-0.42738536f,-0.23324963f,-0.16667747f,
                                           -0.40209758f,-0.75398892f,+0.03567851f};

        tensor_load(x_in, x_in_array);
        tensor_load(hidden_in, hidden_in_array);
        tensor_load(cell_in, cell_in_array);
        tensor_load(hidden_target, hidden_target_array);        
        tensor_load(cell_target, cell_target_array);

        const ulstm_params params = { .num_dims_in = 4,
                                      .num_dims_out = 3,
                                      .W_ih = (float []) {+0.56650645f,-0.24427600f,+0.43296927f,+0.00683678f,
                                                          -0.30415818f,+0.29676661f,-0.30646914f,+0.16980143f,
                                                          -0.16671404f,-0.06329745f,-0.55505770f,-0.27527004f,
                                                          +0.31328988f,-0.14034073f,+0.57506943f,+0.46279728f,
                                                          -0.02703363f,-0.38537154f,+0.35158446f,+0.17918475f,
                                                          -0.37321061f,+0.37501475f,+0.35051039f,+0.51204902f,
                                                          -0.32363924f,-0.09503530f,-0.01118435f,+0.08432633f,
                                                          -0.43816167f,-0.40970066f,+0.31408116f,-0.13538398f,
                                                          +0.28202024f,+0.03291470f,+0.18956997f,+0.12695280f,
                                                          +0.20992422f,+0.28619871f,-0.53469104f,+0.29061010f,
                                                          -0.40594837f,-0.43561596f,+0.03511120f,-0.09838463f,
                                                          +0.33909169f,-0.33436412f,-0.51325393f,+0.42017749f},
                                      .W_hh = (float []) {-0.08561055f,+0.32473481f,+0.18560919f,-0.43294069f,
                                                          +0.11598868f,+0.13867757f,-0.38656986f,-0.27393669f,
                                                          +0.19686237f,+0.10341646f,-0.24558899f,-0.17477310f,
                                                          +0.52878511f,-0.10682208f,+0.32550526f,+0.24999450f,
                                                          -0.37320334f,-0.49099571f,+0.55419201f,+0.03014736f,
                                                          +0.39574939f,+0.11964510f,+0.18568897f,+0.43125674f,
                                                          +0.54745197f,-0.38311866f,+0.07218551f,+0.43085185f,
                                                          +0.41825280f,+0.35867220f,-0.41783929f,-0.41575235f,
                                                          -0.34916505f,+0.07253156f,+0.57542473f,-0.36469808f},
                                      .b_ih = (float []) {+0.30765894f,-0.31955650f,-0.54283381f,-0.12270983f,
                                                          +0.33265242f,+0.53599751f,-0.35855690f,+0.12529148f,
                                                          +0.49817255f,+0.38258320f,+0.35981801f,+0.41028389f},
                                      .b_hh = (float []) {+0.36518350f,+0.14912748f,-0.39480373f,-0.48483479f,
                                                          -0.26455379f,-0.06723007f,-0.35388812f,+0.21122570f,
                                                          +0.17868288f,-0.13073248f,+0.22189924f,+0.18659079f} };

        ulstm * layer = ulstm_construct(&params);

        ulstm_forward(layer, x_in, hidden_in, cell_in, hidden_pred, cell_pred);

        if ((tensors_compare(hidden_pred, hidden_target, eps) != 0) ||
            (tensors_compare(cell_pred, cell_target, eps) != 0)) {
            return -1;
        }

        ulstm_destroy(layer);

        tensor_destroy(x_in);
        tensor_destroy(hidden_in);
        tensor_destroy(cell_in);
        tensor_destroy(hidden_target);
        tensor_destroy(cell_target);
        tensor_destroy(hidden_pred);
        tensor_destroy(cell_pred);

    }

    return 0;

}
