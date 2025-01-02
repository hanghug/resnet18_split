#include <iostream>

#include "resnet18_split3.h"
#include "parameters.h"

// hls-fpga-machine-learning insert weights
#include "weights/w30.h"
#include "weights/b30.h"
#include "weights/w33.h"
#include "weights/b33.h"

// hls-fpga-machine-learning start
void kernel3(
    hls::stream<layer29_t> (&layer29_out)[N_FILT_23/128],
    hls::stream<layer36_t> (&layer36_out)[N_FILT_23/128]
) {
    // hls-fpga-machine-learning insert IO
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight30_t, 147456>(w30, "w30.txt");
        nnet::load_weights_from_txt<bias30_t, 128>(b30, "b30.txt");
        nnet::load_weights_from_txt<weight33_t, 147456>(w33, "w33.txt");
        nnet::load_weights_from_txt<bias33_t, 128>(b33, "b33.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer29_t> layer78_cpy1[N_FILT_23/128];
    #pragma HLS STREAM variable=layer78_cpy1 depth=784
    //#pragma HLS BIND_STORAGE variable=layer78_cpy1 type=fifo impl=srl
    hls::stream<layer29_t> layer78_cpy2[N_FILT_23/128];
    #pragma HLS STREAM variable=layer78_cpy2 depth=784
    //#pragma HLS BIND_STORAGE variable=layer78_cpy2 type=fifo impl=srl
    nnet::clone_stream<layer29_t, layer29_t, 100352, N_FILT_23>(layer29_out, layer78_cpy1, layer78_cpy2); // clone_add_2

    hls::stream<layer90_t> layer90_out[N_CHAN_90/128];
    #pragma HLS STREAM variable=layer90_out depth=900
    //#pragma HLS BIND_STORAGE variable=layer90_out type=fifo impl=srl
    nnet::zeropad2d_cl<layer29_t, layer90_t, config90>(layer78_cpy1, layer90_out); // zp2d_q_conv2d_batchnorm_8

    hls::stream<layer30_t> layer30_out[N_FILT_30/128];
    #pragma HLS STREAM variable=layer30_out depth=784
    //#pragma HLS BIND_STORAGE variable=layer30_out type=fifo impl=srl
    nnet::conv_2d_cl<layer90_t, layer30_t, config30>(layer90_out, layer30_out, w30, b30); // q_conv2d_batchnorm_8

    hls::stream<layer91_t> layer91_out[N_CHAN_91/128];
    #pragma HLS STREAM variable=layer91_out depth=900
    //#pragma HLS BIND_STORAGE variable=layer91_out type=fifo impl=srl
    nnet::zeropad2d_cl<layer32_t, layer91_t, config91>(layer30_out, layer91_out); // zp2d_q_conv2d_batchnorm_9

    hls::stream<layer33_t> layer33_out[N_FILT_33/128];
    #pragma HLS STREAM variable=layer33_out depth=784
    //#pragma HLS BIND_STORAGE variable=layer33_out type=fifo impl=srl
    nnet::conv_2d_cl<layer91_t, layer33_t, config33>(layer91_out, layer33_out, w33, b33); // q_conv2d_batchnorm_9

    nnet::add<layer29_t, layer35_t, layer36_t, config36, N_FILT_23>(layer78_cpy2, layer33_out, layer36_out); // add_3

}
