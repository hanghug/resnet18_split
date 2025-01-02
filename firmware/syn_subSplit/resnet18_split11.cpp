#include <iostream>

#include "resnet18_split11.h"
#include "parameters.h"

// hls-fpga-machine-learning insert weights
#include "weights/w72.h"
#include "weights/b72.h"

// hls-fpga-machine-learning start
void kernel11(
    hls::stream<layer70_t> (&layer70_out)[N_FILT_57/256],
    hls::stream<result_t> (&layer74_out)[N_LAYER_72/200]
) {
    // hls-fpga-machine-learning insert IO
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight72_t, 512000>(w72, "w72.txt");
        nnet::load_weights_from_txt<bias72_t, 1000>(b72, "b72.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer71_t> layer71_out[N_FILT_71/256];
    #pragma HLS STREAM variable=layer71_out depth=1
    // #pragma HLS BIND_STORAGE variable=layer71_out type=fifo impl=srl
    nnet::global_pooling2d_cl<layer70_t, layer71_t, config71>(layer70_out, layer71_out); // global_average_pooling2d

    hls::stream<layer72_t> layer72_out[N_LAYER_72/200];
    #pragma HLS STREAM variable=layer72_out depth=1
    // #pragma HLS BIND_STORAGE variable=layer72_out type=fifo impl=srl
    nnet::dense<layer71_t, layer72_t, config72>(layer71_out, layer72_out, w72, b72); // q_dense

    nnet::softmax<layer72_t, result_t, softmax_config74>(layer72_out, layer74_out); // activation_20

}