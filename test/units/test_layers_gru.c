#include "test_layers_gru.h"

int test_layers_gru(void) {

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
        tensor * hidden = tensor_construct(out_num_dims1, out_num_dims2, out_num_dims3, out_num_dims4);
        tensor * target = tensor_construct(out_num_dims1, out_num_dims2, out_num_dims3, out_num_dims4);
        tensor * pred = tensor_construct(out_num_dims1, out_num_dims2, out_num_dims3, out_num_dims4);

        const float in_array[] = {+1.54099607f,-0.29342890f,-2.17878938f,+0.56843126f,
                                  -1.08452237f,-1.39859545f,+0.40334684f,+0.83802634f};
        const float hidden_array[] = {-0.71925759f,-0.40334353f,-0.59663534f,
                                      +0.18203649f,-0.85667461f,+1.10060418f};
        const float target_array[] = {-0.75540549f,+0.31993112f,+0.16078511f,
                                      -0.02101753f,-0.81274438f,+0.93186295f};

        tensor_load(in, in_array);
        tensor_load(hidden, hidden_array);
        tensor_load(target, target_array);

        const ugru_params params = { .num_dims_in = 4,
                                     .num_dims_out = 3,
                                     .W_ih = (float []) {-0.36346388f,-0.14616501f,-0.22505111f,+0.49883109f,
                                                         -0.37422669f,-0.26577330f,-0.40336025f,-0.54072374f,
                                                         -0.33702272f,+0.49628916f,+0.25762430f,+0.27982581f,
                                                         +0.03036375f,-0.29599795f,+0.09767882f,-0.53906888f,
                                                         -0.41717380f,-0.29764137f,+0.36427218f,+0.33851272f,
                                                         -0.25605199f,-0.02083218f,+0.36925054f,+0.57396299f,
                                                         +0.22914003f,+0.07799590f,+0.38710546f,-0.33994517f,
                                                         +0.10758577f,-0.44762284f,-0.40015346f,-0.29824966f,
                                                         +0.26123542f,+0.23218742f,-0.34199488f,+0.17442161f},
                                     .W_hh = (float []) {+0.31694913f,-0.07287163f,+0.02204412f,+0.13377477f,
                                                         +0.35817459f,+0.55436832f,-0.44491971f,-0.21158139f,
                                                         +0.22690436f,+0.47836322f,+0.50241441f,+0.50942892f,
                                                         +0.11490110f,-0.50205380f,+0.05311189f,-0.36119342f,
                                                         -0.53806394f,+0.51297134f,+0.43899390f,-0.57592303f,
                                                         +0.10806383f,-0.09726043f,-0.09500942f,-0.26428604f,
                                                         +0.22202361f,-0.34196660f,+0.21165161f},
                                     .b_ih = (float []) {+0.29197070f,+0.41330865f,+0.21587770f,-0.57142389f,
                                                         -0.37452531f,+0.28827965f,+0.12084019f,-0.45038170f,
                                                         -0.33244953f},
                                     .b_hh = (float []) {+0.54314184f,+0.38902894f,-0.25173923f,-0.14531028f,
                                                         -0.54998279f,-0.01037737f,-0.43477875f,-0.44534299f,
                                                         -0.03181177f} };

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
        tensor * hidden = tensor_construct(out_num_dims1, out_num_dims2, out_num_dims3, out_num_dims4);
        tensor * target = tensor_construct(out_num_dims1, out_num_dims2, out_num_dims3, out_num_dims4);
        tensor * pred = tensor_construct(out_num_dims1, out_num_dims2, out_num_dims3, out_num_dims4);

        const float in_array[] = {+0.66135216f,+0.26692411f,+0.06167726f,+0.62131733f,
                                  -0.45190597f,-0.16613023f,-1.52276850f,+0.38168392f};
        const float hidden_array[] = {-1.02760863f,-0.56305277f,-0.89229053f,
                                      -0.05825018f,-0.19550958f,-0.96563596f};
        const float target_array[] = {-0.16974396f,-0.23915966f,-0.32620773f,
                                      -0.40911978f,-0.38766432f,-0.79748154f};

        tensor_load(in, in_array);
        tensor_load(hidden, hidden_array);
        tensor_load(target, target_array);

        const ugru_params params = { .num_dims_in = 4,
                                     .num_dims_out = 3,
                                     .W_ih = (float []) {-0.56438923f,+0.35791126f,+0.16129081f,+0.54764873f,
                                                         +0.38108569f,-0.52604556f,-0.54894948f,-0.27847457f,
                                                         +0.50697803f,-0.09616865f,+0.24708250f,-0.26830125f,
                                                         +0.56650645f,-0.24427600f,+0.43296927f,+0.00683678f,
                                                         -0.30415818f,+0.29676661f,-0.30646914f,+0.16980143f,
                                                         -0.16671404f,-0.06329745f,-0.55505770f,-0.27527004f,
                                                         +0.31328988f,-0.14034073f,+0.57506943f,+0.46279728f,
                                                         -0.02703363f,-0.38537154f,+0.35158446f,+0.17918475f,
                                                         -0.37321061f,+0.37501475f,+0.35051039f,+0.51204902f},
                                     .W_hh = (float []) {-0.32363924f,-0.09503530f,-0.01118435f,+0.08432633f,
                                                         -0.43816167f,-0.40970066f,+0.31408116f,-0.13538398f,
                                                         +0.28202024f,+0.03291470f,+0.18956997f,+0.12695280f,
                                                         +0.20992422f,+0.28619871f,-0.53469104f,+0.29061010f,
                                                         -0.40594837f,-0.43561596f,+0.03511120f,-0.09838463f,
                                                         +0.33909169f,-0.33436412f,-0.51325393f,+0.42017749f,
                                                         -0.08561055f,+0.32473481f,+0.18560919f},
                                     .b_ih = (float []) {-0.43294069f,+0.11598868f,+0.13867757f,-0.38656986f,
                                                         -0.27393669f,+0.19686237f,+0.10341646f,-0.24558899f,
                                                         -0.17477310f},
                                     .b_hh = (float []) {+0.52878511f,-0.10682208f,+0.32550526f,+0.24999450f,
                                                         -0.37320334f,-0.49099571f,+0.55419201f,+0.03014736f,
                                                         +0.39574939f} };

        ugru * layer = ugru_construct(&params);

        ugru_forward(layer, in, hidden, pred);

        if (tensors_compare(pred, target, eps) != 0) {
            return -2;
        }

        tensor_destroy(in);
        tensor_destroy(hidden);
        tensor_destroy(target);
        tensor_destroy(pred);

    }

    return 0;

}