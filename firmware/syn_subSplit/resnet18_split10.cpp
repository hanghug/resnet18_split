#include <iostream>

#include "resnet18_split10.h"
#include "parameters.h"

// hls-fpga-machine-learning insert weights
#include "weights/w67_1_1.h"
#include "weights/w67_2_1.h"
#include "weights/b67.h"
#include "weights/b67_0.h"

// hls-fpga-machine-learning start
void kernel10(
    hls::stream<layer66_t> (&layer66_out)[N_FILT_64/256], hls::stream<layer63_t> (&layer82_cpy2)[N_FILT_57/256],
    hls::stream<layer70_t> (&layer70_out)[N_FILT_57/256]
) {
    // hls-fpga-machine-learning insert IO
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight64_t, 2359296/2>(w67_1_1, "w67_1_1.txt");
        nnet::load_weights_from_txt<weight64_t, 2359296/2>(w67_2_1, "w67_2_1.txt");
        nnet::load_weights_from_txt<bias64_t, 512>(b67, "b67.txt");
        nnet::load_weights_from_txt<bias64_t, 512>(b67_0, "b67_0.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer99_t> layer99_out[N_CHAN_99/256];
    #pragma HLS STREAM variable=layer99_out depth=81
    // #pragma HLS BIND_STORAGE variable=layer99_out type=fifo impl=srl
    nnet::zeropad2d_cl<layer66_t, layer99_t, config99>(layer66_out, layer99_out); // zp2d_q_conv2d_batchnorm_19


    hls::stream<layer75_t> layer75_out1[N_FILT_75/256];
    #pragma HLS STREAM variable=layer75_out1 depth=81
    #pragma HLS BIND_STORAGE variable=layer75_out1 type=fifo impl=srl
    hls::stream<layer75_t> layer75_out2[N_FILT_75/256];
    #pragma HLS STREAM variable=layer75_out2 depth=81
    #pragma HLS BIND_STORAGE variable=layer75_out2 type=fifo impl=srl
    nnet::split2<layer99_t, layer75_t, config67_split>(layer99_out, layer75_out1, layer75_out2);


    hls::stream<layer67_t> layer67_out1[N_FILT_67/256];
    #pragma HLS STREAM variable=layer67_out1 depth=49
    #pragma HLS BIND_STORAGE variable=layer67_out1 type=fifo impl=srl
    nnet::conv_2d_cl<layer75_t, layer67_t, config67_1>(layer75_out1, layer67_out1, w67_1_1, b67); // q_conv2d_batchnorm_19_1

    hls::stream<layer67_t> layer67_out2[N_FILT_67/256];
    #pragma HLS STREAM variable=layer67_out2 depth=49
    #pragma HLS BIND_STORAGE variable=layer67_out2 type=fifo impl=srl
    nnet::conv_2d_cl<layer75_t, layer67_t, config67_2>(layer75_out2, layer67_out2, w67_2_1, b67_0); // q_conv2d_batchnorm_19_1


    hls::stream<layer67_t> layer67_out[N_FILT_67/256];
    #pragma HLS STREAM variable=layer67_out depth=49
    nnet::add2<layer67_t, config67_add>(layer67_out1,layer67_out2, layer67_out); 

    nnet::add<layer63_t, layer69_t, layer70_t, config70, N_FILT_57>(layer82_cpy2, layer67_out, layer70_out); // add_7

}