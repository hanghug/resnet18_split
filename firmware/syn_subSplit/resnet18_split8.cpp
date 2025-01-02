#include <iostream>

#include "resnet18_split8.h"
#include "parameters.h"

// hls-fpga-machine-learning insert weights
#include "weights/w102.h"
#include "weights/b102.h"

// hls-fpga-machine-learning start
void kernel8(
    hls::stream<layer53_t> (&layer81_cpy2)[N_FILT_40/256],
    hls::stream<layer61_t> (&layer61_out)[N_FILT_57/256]
) {
    // hls-fpga-machine-learning insert IO
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight102_t, 131072>(w102, "w102.txt");
        nnet::load_weights_from_txt<bias102_t, 512>(b102, "b102.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    nnet::pointwise_conv_2d_cl<layer53_t, layer102_t, config102>(layer81_cpy2, layer61_out, w102, b102); // q_conv2d_batchnorm_17

}
