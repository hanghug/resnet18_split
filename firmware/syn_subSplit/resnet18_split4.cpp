#include <iostream>

#include "resnet18_split4.h"
#include "parameters.h"

// hls-fpga-machine-learning insert weights
#include "weights/w37.h"
#include "weights/b37.h"
#include "weights/w42.h"
#include "weights/b42.h"
#include "weights/w101.h"
#include "weights/b101.h"

// hls-fpga-machine-learning start
void kernel4(
    hls::stream<layer36_t> (&layer36_out)[N_FILT_23/128],
    hls::stream<layer46_t> (&layer46_out)[N_FILT_40/256]
) {
    // hls-fpga-machine-learning insert IO
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight37_t, 294912>(w37, "w37.txt");
        nnet::load_weights_from_txt<bias37_t, 256>(b37, "b37.txt");
        nnet::load_weights_from_txt<weight42_t, 589824>(w42, "w42.txt");
        nnet::load_weights_from_txt<bias42_t, 256>(b42, "b42.txt");
        nnet::load_weights_from_txt<weight101_t, 32768>(w101, "w101.txt");
        nnet::load_weights_from_txt<bias101_t, 256>(b101, "b101.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer36_t> layer79_cpy1[N_FILT_23/128];
    #pragma HLS STREAM variable=layer79_cpy1 depth=784
    //#pragma HLS BIND_STORAGE variable=layer79_cpy1 type=fifo impl=srl
    hls::stream<layer36_t> layer79_cpy2[N_FILT_23/128];
    #pragma HLS STREAM variable=layer79_cpy2 depth=784
    //#pragma HLS BIND_STORAGE variable=layer79_cpy2 type=fifo impl=srl
    nnet::clone_stream<layer36_t, layer36_t, 100352, N_FILT_23>(layer36_out, layer79_cpy1, layer79_cpy2); // clone_add_3

    hls::stream<layer92_t> layer92_out[N_CHAN_92/128];
    #pragma HLS STREAM variable=layer92_out depth=841
    //#pragma HLS BIND_STORAGE variable=layer92_out type=fifo impl=srl
    nnet::zeropad2d_cl<layer36_t, layer92_t, config92>(layer79_cpy1, layer92_out); // zp2d_q_conv2d_batchnorm_10

    hls::stream<layer37_t> layer37_out[N_FILT_37/256];
    #pragma HLS STREAM variable=layer37_out depth=196
    //#pragma HLS BIND_STORAGE variable=layer37_out type=fifo impl=srl
    nnet::conv_2d_cl<layer92_t, layer37_t, config37>(layer92_out, layer37_out, w37, b37); // q_conv2d_batchnorm_10

    hls::stream<layer93_t> layer93_out[N_CHAN_93/256];
    #pragma HLS STREAM variable=layer93_out depth=256
    //#pragma HLS BIND_STORAGE variable=layer93_out type=fifo impl=srl
    nnet::zeropad2d_cl<layer39_t, layer93_t, config93>(layer37_out, layer93_out); // zp2d_q_conv2d_batchnorm_11

    hls::stream<layer42_t> layer42_out[N_FILT_42/256];
    #pragma HLS STREAM variable=layer42_out depth=196
    //#pragma HLS BIND_STORAGE variable=layer42_out type=fifo impl=srl
    nnet::conv_2d_cl<layer93_t, layer42_t, config42>(layer93_out, layer42_out, w42, b42); // q_conv2d_batchnorm_11

    hls::stream<layer101_t> layer101_out[N_FILT_101/256];
    #pragma HLS STREAM variable=layer101_out depth=196
    //#pragma HLS BIND_STORAGE variable=layer101_out type=fifo impl=srl
    nnet::pointwise_conv_2d_cl<layer36_t, layer101_t, config101>(layer79_cpy2, layer101_out, w101, b101); // q_conv2d_batchnorm_12

    nnet::add<layer44_t, layer45_t, layer46_t, config46, N_FILT_40>(layer101_out, layer42_out, layer46_out); // add_4

}
