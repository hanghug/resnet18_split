#include <iostream>

#include "resnet18_split1.h"
#include "parameters.h"

// hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/w6.h"
#include "weights/b6.h"
#include "weights/w9.h"
#include "weights/b9.h"

// hls-fpga-machine-learning start
void kernel1(
    hls::stream<input_t> (&input_1)[N_INPUT_3_1/3],
    hls::stream<layer12_t> (&layer12_out)[N_FILT_5/64]
) {
    // hls-fpga-machine-learning insert IO
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight2_t, 9408>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 64>(b2, "b2.txt");
        nnet::load_weights_from_txt<weight6_t, 36864>(w6, "w6.txt");
        nnet::load_weights_from_txt<bias6_t, 64>(b6, "b6.txt");
        nnet::load_weights_from_txt<weight9_t, 36864>(w9, "w9.txt");
        nnet::load_weights_from_txt<bias9_t, 64>(b9, "b9.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer83_t> layer83_out[N_CHAN_83/3];
    #pragma HLS STREAM variable=layer83_out depth=1024
    //#pragma HLS BIND_STORAGE variable=layer83_out type=fifo impl=srl    //size of variable 'SRL_SIG' is too large to handle; the size of the variable is 1258584, the limit is 1000000.
    nnet::zeropad2d_cl<input_t, layer83_t, config83>(input_1, layer83_out); // zp2d_q_conv2d_batchnorm

    hls::stream<layer2_t> layer2_out[N_FILT_2/64];
    #pragma HLS STREAM variable=layer2_out depth=1024
    //#pragma HLS BIND_STORAGE variable=layer2_out type=fifo impl=srl    //size of variable 'SRL_SIG' is too large to handle; the size of the variable is 6422528, the limit is 1000000.
    nnet::conv_2d_cl<layer83_t, layer2_t, config2>(layer83_out, layer2_out, w2, b2); // q_conv2d_batchnorm

    hls::stream<layer5_t> layer5_out[N_FILT_5/64];
    #pragma HLS STREAM variable=layer5_out depth=1024
    //#pragma HLS BIND_STORAGE variable=layer5_out type=fifo impl=srl    //size of variable 'SRL_SIG' is too large to handle; the size of the variable is 1605632, the limit is 1000000.
    nnet::pooling2d_cl<layer4_t, layer5_t, config5>(layer2_out, layer5_out); // max_pooling2d

    hls::stream<layer5_t> layer75_cpy1[N_FILT_5/64];
    #pragma HLS STREAM variable=layer75_cpy1 depth=1024
    //#pragma HLS BIND_STORAGE variable=layer75_cpy1 type=fifo impl=srl    //size of variable 'SRL_SIG' is too large to handle; the size of the variable is 1605632, the limit is 1000000.
    hls::stream<layer5_t> layer75_cpy2[N_FILT_5/64];
    #pragma HLS STREAM variable=layer75_cpy2 depth=1024
    //#pragma HLS BIND_STORAGE variable=layer75_cpy2 type=fifo impl=srl    //size of variable 'SRL_SIG' is too large to handle; the size of the variable is 1605632, the limit is 1000000.
    nnet::clone_stream<layer5_t, layer5_t, 200704, N_FILT_5>(layer5_out, layer75_cpy1, layer75_cpy2); // clone_max_pooling2d

    hls::stream<layer84_t> layer84_out[N_CHAN_84/64];
    #pragma HLS STREAM variable=layer84_out depth=1024
    //#pragma HLS BIND_STORAGE variable=layer84_out type=fifo impl=srl    //size of variable 'SRL_SIG' is too large to handle; the size of the variable is 1722368, the limit is 1000000.
    nnet::zeropad2d_cl<layer5_t, layer84_t, config84>(layer75_cpy1, layer84_out); // zp2d_q_conv2d_batchnorm_1

    hls::stream<layer6_t> layer6_out[N_FILT_6/64];
    #pragma HLS STREAM variable=layer6_out depth=1024
    //#pragma HLS BIND_STORAGE variable=layer6_out type=fifo impl=srl    //size of variable 'SRL_SIG' is too large to handle; the size of the variable is 1605632, the limit is 1000000.
    nnet::conv_2d_cl<layer84_t, layer6_t, config6>(layer84_out, layer6_out, w6, b6); // q_conv2d_batchnorm_1

    hls::stream<layer85_t> layer85_out[N_CHAN_85/64];
    #pragma HLS STREAM variable=layer85_out depth=1024
    //#pragma HLS BIND_STORAGE variable=layer85_out type=fifo impl=srl    //size of variable 'SRL_SIG' is too large to handle; the size of the variable is 1722368, the limit is 1000000.
    nnet::zeropad2d_cl<layer8_t, layer85_t, config85>(layer6_out, layer85_out); // zp2d_q_conv2d_batchnorm_2

    hls::stream<layer9_t> layer9_out[N_FILT_9/64];
    #pragma HLS STREAM variable=layer9_out depth=1024
    //#pragma HLS BIND_STORAGE variable=layer9_out type=fifo impl=srl    //size of variable 'SRL_SIG' is too large to handle; the size of the variable is 1605632, the limit is 1000000.
    nnet::conv_2d_cl<layer85_t, layer9_t, config9>(layer85_out, layer9_out, w9, b9); // q_conv2d_batchnorm_2

    nnet::add<layer5_t, layer11_t, layer12_t, config12, N_FILT_5>(layer75_cpy2, layer9_out, layer12_out); // add

}