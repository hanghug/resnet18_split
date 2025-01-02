#include <iostream>

#include "resnet18_split9.h"
#include "parameters.h"

// hls-fpga-machine-learning insert weights
#include "weights/w64_1_1.h"
#include "weights/w64_2_1.h"
#include "weights/b64.h"
#include "weights/b64_0.h"

// hls-fpga-machine-learning start
void kernel9(
    hls::stream<layer61_t> (&layer61_out)[N_FILT_57/256], hls::stream<layer62_t> (&layer62_out)[N_FILT_59/256],
    hls::stream<layer66_t> (&layer66_out)[N_FILT_64/256], hls::stream<layer63_t> (&layer82_cpy2)[N_FILT_57/256]
) {
    // hls-fpga-machine-learning insert IO
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight64_t, 2359296/2>(w64_1_1, "w64_1_1.txt");
        nnet::load_weights_from_txt<weight64_t, 2359296/2>(w64_2_1, "w64_2_1.txt");
        nnet::load_weights_from_txt<bias64_t, 512>(b64, "b64.txt");
        nnet::load_weights_from_txt<bias64_t, 512>(b64_0, "b64_0.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer63_t> layer63_out[N_FILT_57/256];
    #pragma HLS STREAM variable=layer63_out depth=49
    #pragma HLS BIND_STORAGE variable=layer63_out type=fifo impl=srl
    nnet::add<layer61_t, layer62_t, layer63_t, config63, N_FILT_57>(layer61_out, layer62_out, layer63_out); // add_6

    hls::stream<layer63_t> layer82_cpy1[N_FILT_57/256];
    #pragma HLS STREAM variable=layer82_cpy1 depth=49
    #pragma HLS BIND_STORAGE variable=layer82_cpy1 type=fifo impl=srl
    nnet::clone_stream<layer63_t, layer63_t, 25088, N_FILT_57>(layer63_out, layer82_cpy1, layer82_cpy2); // clone_add_6

    hls::stream<layer98_t> layer98_out[N_CHAN_98/256];
    #pragma HLS STREAM variable=layer98_out depth=81
    #pragma HLS BIND_STORAGE variable=layer98_out type=fifo impl=srl
    nnet::zeropad2d_cl<layer63_t, layer98_t, config98>(layer82_cpy1, layer98_out); // zp2d_q_conv2d_batchnorm_18


    hls::stream<layer74_t> layer74_out1[N_FILT_74/256];
    #pragma HLS STREAM variable=layer74_out1 depth=81
    hls::stream<layer74_t> layer74_out2[N_FILT_74/256];
    #pragma HLS STREAM variable=layer74_out2 depth=81
    nnet::split2<layer98_t, layer74_t, config64_split>(layer98_out, layer74_out1, layer74_out2);


    hls::stream<layer66_t> layer66_out1[N_FILT_64/256];
    #pragma HLS STREAM variable=layer66_out1 depth=49
    nnet::conv_2d_cl<layer74_t, layer66_t, config64_1>(layer74_out1, layer66_out1, w64_1_1, b64); // q_conv2d_batchnorm_18_1

    hls::stream<layer66_t> layer66_out2[N_FILT_64/256];
    #pragma HLS STREAM variable=layer66_out2 depth=49
    nnet::conv_2d_cl<layer74_t, layer66_t, config64_2>(layer74_out2, layer66_out2, w64_2_1, b64_0); // q_conv2d_batchnorm_18_2

    nnet::add2<layer66_t, config64_add>(layer66_out1, layer66_out2, layer66_out);
}
