#include <iostream>

#include "resnet18_split5.h"
#include "parameters.h"

// hls-fpga-machine-learning insert weights
#include "weights/w47.h"
#include "weights/b47.h"
#include "weights/w50.h"
#include "weights/b50.h"

// hls-fpga-machine-learning start
void kernel5(
    hls::stream<layer46_t> (&layer46_out)[N_FILT_40/256],
    hls::stream<layer53_t> (&layer53_out)[N_FILT_40/256]
) {
    // hls-fpga-machine-learning insert IO
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight47_t, 589824>(w47, "w47.txt");
        nnet::load_weights_from_txt<bias47_t, 256>(b47, "b47.txt");
        nnet::load_weights_from_txt<weight50_t, 589824>(w50, "w50.txt");
        nnet::load_weights_from_txt<bias50_t, 256>(b50, "b50.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer46_t> layer80_cpy1[N_FILT_40/256];
    #pragma HLS STREAM variable=layer80_cpy1 depth=196
    //#pragma HLS BIND_STORAGE variable=layer80_cpy1 type=fifo impl=srl
    hls::stream<layer46_t> layer80_cpy2[N_FILT_40/256];
    #pragma HLS STREAM variable=layer80_cpy2 depth=196
    //#pragma HLS BIND_STORAGE variable=layer80_cpy2 type=fifo impl=srl
    nnet::clone_stream<layer46_t, layer46_t, 50176, N_FILT_40>(layer46_out, layer80_cpy1, layer80_cpy2); // clone_add_4

    hls::stream<layer94_t> layer94_out[N_CHAN_94/256];
    #pragma HLS STREAM variable=layer94_out depth=256
    //#pragma HLS BIND_STORAGE variable=layer94_out type=fifo impl=srl
    nnet::zeropad2d_cl<layer46_t, layer94_t, config94>(layer80_cpy1, layer94_out); // zp2d_q_conv2d_batchnorm_13

    hls::stream<layer47_t> layer47_out[N_FILT_47/256];
    #pragma HLS STREAM variable=layer47_out depth=196
    //#pragma HLS BIND_STORAGE variable=layer47_out type=fifo impl=srl
    nnet::conv_2d_cl<layer94_t, layer47_t, config47>(layer94_out, layer47_out, w47, b47); // q_conv2d_batchnorm_13

    hls::stream<layer95_t> layer95_out[N_CHAN_95/256];
    #pragma HLS STREAM variable=layer95_out depth=256
    //#pragma HLS BIND_STORAGE variable=layer95_out type=fifo impl=srl
    nnet::zeropad2d_cl<layer49_t, layer95_t, config95>(layer47_out, layer95_out); // zp2d_q_conv2d_batchnorm_14

    hls::stream<layer50_t> layer50_out[N_FILT_50/256];
    #pragma HLS STREAM variable=layer50_out depth=196
    //#pragma HLS BIND_STORAGE variable=layer50_out type=fifo impl=srl
    nnet::conv_2d_cl<layer95_t, layer50_t, config50>(layer95_out, layer50_out, w50, b50); // q_conv2d_batchnorm_14

    nnet::add<layer46_t, layer52_t, layer53_t, config53, N_FILT_40>(layer80_cpy2, layer50_out, layer53_out); // add_5

}

