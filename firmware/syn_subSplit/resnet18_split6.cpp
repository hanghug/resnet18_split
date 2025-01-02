#include <iostream>

#include "resnet18_split6.h"
#include "parameters.h"

// hls-fpga-machine-learning insert weights
#include "weights/w54.h"
#include "weights/b54.h"

// hls-fpga-machine-learning start
void kernel6(
    hls::stream<layer53_t> (&layer53_out)[N_FILT_40/256],
    hls::stream<layer56_t> (&layer56_out)[N_FILT_54/256], hls::stream<layer53_t> (&layer81_cpy2)[N_FILT_40/256]
) {
    // hls-fpga-machine-learning insert IO
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight54_t, 1179648>(w54, "w54.txt");
        nnet::load_weights_from_txt<bias54_t, 512>(b54, "b54.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer53_t> layer81_cpy1[N_FILT_40/256];
    #pragma HLS STREAM variable=layer81_cpy1 depth=196
    //#pragma HLS BIND_STORAGE variable=layer81_cpy1 type=fifo impl=srl
    nnet::clone_stream<layer53_t, layer53_t, 50176, N_FILT_40>(layer53_out, layer81_cpy1, layer81_cpy2); // clone_add_5

    hls::stream<layer96_t> layer96_out[N_CHAN_96/256];
    #pragma HLS STREAM variable=layer96_out depth=225
    //#pragma HLS BIND_STORAGE variable=layer96_out type=fifo impl=srl
    nnet::zeropad2d_cl<layer53_t, layer96_t, config96>(layer81_cpy1, layer96_out); // zp2d_q_conv2d_batchnorm_15

    nnet::conv_2d_cl<layer96_t, layer54_t, config54>(layer96_out, layer56_out, w54, b54); // q_conv2d_batchnorm_15

}
