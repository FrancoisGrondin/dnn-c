#ifndef __LAYERS_LSTM
#define __LAYERS_LSTM

#include "../utils/tensor.h"

typedef struct ulstm_params {

    unsigned int num_dims_in;  // I
    unsigned int num_dims_out; // O

    float * W_ih; // 4O x I
    float * W_hh; // 4O x O
    float * b_ih; // 4O x 1
    float * b_hh; // 4O x 1

} ulstm_params;

typedef struct ulstm {

    const ulstm_params * params;

    float * h;
    float * c;

} ulstm;

ulstm * ulstm_construct(const ulstm_params * params);

void ulstm_destroy(ulstm * obj);

int ulstm_forward(const ulstm * obj, const tensor * in, const tensor * hidden_in, const tensor * cell_in, tensor * hidden_out, tensor * cell_out);

#endif // __LAYERS_LSTM