#include <layers/gru.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

ugru * ugru_construct(const ugru_params * params) {

    ugru * obj = (ugru *) malloc(sizeof(ugru));

    obj->params = params;

    obj->h = (float *) calloc(sizeof(float), params->num_dims_out);

    return obj;

}

void ugru_destroy(ugru * obj) {

    free((void *) obj->h);

    free((void *) obj);

}

int ugru_forward(const ugru * obj, const tensor * in, const tensor * hidden, tensor * out) {

    const unsigned int num_dims_batch = in->num_dims1 * in->num_dims2 * in->num_dims3;
    const unsigned int num_dims_in = obj->params->num_dims_in;
    const unsigned int num_dims_out = obj->params->num_dims_out;

    for (unsigned int index_dim_batch = 0; index_dim_batch < num_dims_batch; index_dim_batch++) {

        for (unsigned int index_dim_out = 0; index_dim_out < num_dims_out; index_dim_out++) {

            float b_ir = obj->params->b_ir[index_dim_out];
            float b_hr = obj->params->b_hr[index_dim_out];

            float b_iz = obj->params->b_iz[index_dim_out];
            float b_hz = obj->params->b_hz[index_dim_out];

            float b_in = obj->params->b_in[index_dim_out];
            float b_hn = obj->params->b_hn[index_dim_out];

            float r_arg = b_ir + b_hr;
            float z_arg = b_iz + b_hz;
            float n_arg1 = b_in;
            float n_arg2 = b_hn;

            for (unsigned int index_dim_in = 0; index_dim_in < num_dims_in; index_dim_in++) {

                float w_ir = obj->params->W_ir[index_dim_out * num_dims_in + index_dim_in];
                float w_iz = obj->params->W_iz[index_dim_out * num_dims_in + index_dim_in];
                float w_in = obj->params->W_in[index_dim_out * num_dims_in + index_dim_in];

                float x = in->data[index_dim_batch * num_dims_in + index_dim_in];
                r_arg += w_ir * x;
                z_arg += w_iz * x;
                n_arg1 += w_in * x;

            }

            for (unsigned int index_dim_hidden = 0; index_dim_hidden < num_dims_out; index_dim_hidden++) {

                float w_hr = obj->params->W_hr[index_dim_out * num_dims_out + index_dim_hidden];
                float w_hz = obj->params->W_hz[index_dim_out * num_dims_out + index_dim_hidden];
                float w_hn = obj->params->W_hn[index_dim_out * num_dims_out + index_dim_hidden];

                float h = hidden->data[index_dim_batch * num_dims_out + index_dim_hidden];
                r_arg += w_hr * h;
                z_arg += w_hz * h;
                n_arg2 += w_hn * h;

            }

            float r = 1.0f / (1.0f + expf(-1.0f * r_arg));
            float z = 1.0f / (1.0f + expf(-1.0f * z_arg));
            float n = tanhf(n_arg1 + r * n_arg2);
            obj->h[index_dim_out] = (1.0f - z) * n + z * hidden->data[index_dim_batch * num_dims_out + index_dim_out];

        }

        memcpy(&(hidden->data[index_dim_batch * num_dims_out]), obj->h, sizeof(float) * num_dims_out);

    }   

    return 0;

}
