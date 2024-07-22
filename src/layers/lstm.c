#include <layers/lstm.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

ulstm * ulstm_construct(const ulstm_params * params) {

    ulstm * obj = (ulstm *) malloc(sizeof(ulstm));

    obj->params = params;

    obj->h = (float *) calloc(sizeof(float), params->num_dims_out);
    obj->c = (float *) calloc(sizeof(float), params->num_dims_out);

    return obj;

}

void ulstm_destroy(ulstm * obj) {

    free((void *) obj->h);
    free((void *) obj->c);

    free((void *) obj);	

}

int ulstm_forward(const ulstm * obj, const tensor * in, const tensor * hidden_in, const tensor * cell_in, tensor * hidden_out, tensor * cell_out) {

    const unsigned int num_dims_batch = in->num_dims1 * in->num_dims2 * in->num_dims3;
    const unsigned int num_dims_in = obj->params->num_dims_in;
    const unsigned int num_dims_out = obj->params->num_dims_out;

    // Parameters are saved:
    //
    // W_ih = [ W_ii | W_if | W_ig | W_io ]
    // W_hh = [ W_hi | W_hf | W_hg | W_ho ]
    //
    // b_ih = [ b_ii | b_if | b_ig | b_io ]
    // b_hh = [ b_hi | b_hf | b_hg | b_ho ]
    //
    // And inference goes this way:
    //
    // i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi})
    // f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf})
    // g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{(t-1)} + b_{hg})
    // o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho})
    // c_t = f_t \odot c_{(t-1)} + i_t \odot g_t
    // h_t = o_t \odot \tanh(c_t)

    float * Wm_ii = &(obj->params->W_ih[0 * num_dims_in * num_dims_out]);
    float * Wm_if = &(obj->params->W_ih[1 * num_dims_in * num_dims_out]);
    float * Wm_ig = &(obj->params->W_ih[2 * num_dims_in * num_dims_out]);
    float * Wm_io = &(obj->params->W_ih[3 * num_dims_in * num_dims_out]);
    float * Wm_hi = &(obj->params->W_hh[0 * num_dims_out * num_dims_out]);
    float * Wm_hf = &(obj->params->W_hh[1 * num_dims_out * num_dims_out]);
    float * Wm_hg = &(obj->params->W_hh[2 * num_dims_out * num_dims_out]);
    float * Wm_ho = &(obj->params->W_hh[3 * num_dims_out * num_dims_out]);
    float * bv_ii = &(obj->params->b_ih[0 * num_dims_out]);
    float * bv_if = &(obj->params->b_ih[1 * num_dims_out]);
    float * bv_ig = &(obj->params->b_ih[2 * num_dims_out]);
    float * bv_io = &(obj->params->b_ih[3 * num_dims_out]);
    float * bv_hi = &(obj->params->b_hh[0 * num_dims_out]);
    float * bv_hf = &(obj->params->b_hh[1 * num_dims_out]);
    float * bv_hg = &(obj->params->b_hh[2 * num_dims_out]);
    float * bv_ho = &(obj->params->b_hh[3 * num_dims_out]);    

    for (unsigned int index_dim_batch = 0; index_dim_batch < num_dims_batch; index_dim_batch++) {

        float * xv = &(in->data[index_dim_batch * num_dims_in]);
        float * hv = &(hidden_in->data[index_dim_batch * num_dims_out]);
        float * cv = &(cell_in->data[index_dim_batch * num_dims_out]);

        for (unsigned int index_dim_out = 0; index_dim_out < num_dims_out; index_dim_out++) {

            float b_ii = bv_ii[index_dim_out];
            float b_hi = bv_hi[index_dim_out];

            float b_if = bv_if[index_dim_out];
            float b_hf = bv_hf[index_dim_out];

            float b_ig = bv_ig[index_dim_out];
            float b_hg = bv_hg[index_dim_out];

            float b_io = bv_io[index_dim_out];
            float b_ho = bv_ho[index_dim_out];

            float i_arg = b_ii + b_hi;
            float f_arg = b_if + b_hf;
            float g_arg = b_ig + b_hg;
            float o_arg = b_io + b_ho;

            for (unsigned int index_dim_in = 0; index_dim_in < num_dims_in; index_dim_in++) {

                float W_ii = Wm_ii[index_dim_out * num_dims_in + index_dim_in];
                float W_if = Wm_if[index_dim_out * num_dims_in + index_dim_in];
                float W_ig = Wm_ig[index_dim_out * num_dims_in + index_dim_in];
                float W_io = Wm_io[index_dim_out * num_dims_in + index_dim_in];

                float x = xv[index_dim_in];

                i_arg += W_ii * x;
                f_arg += W_if * x;
                g_arg += W_ig * x;
                o_arg += W_io * x;

            }

            for (unsigned int index_dim_hidden = 0; index_dim_hidden < num_dims_out; index_dim_hidden++) {

                float W_hi = Wm_hi[index_dim_out * num_dims_out + index_dim_hidden];
                float W_hf = Wm_hf[index_dim_out * num_dims_out + index_dim_hidden];
                float W_hg = Wm_hg[index_dim_out * num_dims_out + index_dim_hidden];
                float W_ho = Wm_ho[index_dim_out * num_dims_out + index_dim_hidden];

                float h = hv[index_dim_hidden];

                i_arg += W_hi * h;
                f_arg += W_hf * h;
                g_arg += W_hg * h;
                o_arg += W_ho * h;

            }

            float i = 1.0f / (1.0f + expf(-1.0f * i_arg));
            float f = 1.0f / (1.0f + expf(-1.0f * f_arg));
            float g = tanhf(g_arg);
            float o = 1.0f / (1.0f + expf(-1.0f * o_arg));

            obj->c[index_dim_out] = f * cv[index_dim_out] + i * g;
            obj->h[index_dim_out] = o * tanhf(obj->c[index_dim_out]);

        }

        memcpy(&(hidden_out->data[index_dim_batch * num_dims_out]), obj->h, sizeof(float) * num_dims_out);
        memcpy(&(cell_out->data[index_dim_batch * num_dims_out]), obj->c, sizeof(float) * num_dims_out);

    }   


}