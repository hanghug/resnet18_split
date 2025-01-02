#include <iostream>

#include "resnet18_split7.h"
#include "parameters.h"

// hls-fpga-machine-learning insert weights
#include "weights/w59_1_1.h"
#include "weights/w59_2_1.h"
#include "weights/b59.h"
#include "weights/b59_0.h"

// hls-fpga-machine-learning start
void kernel7(
    hls::stream<layer56_t> (&layer56_out)[N_FILT_54/256],
    hls::stream<layer62_t> (&layer62_out)[N_FILT_59/256]
) {
    // hls-fpga-machine-learning insert IO
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight59_t, 2359296/2>(w59_1_1, "w59_1_1.txt");
        nnet::load_weights_from_txt<weight59_t, 2359296/2>(w59_2_1, "w59_2_1.txt");
        nnet::load_weights_from_txt<bias59_t, 512>(b59, "b59.txt");
        nnet::load_weights_from_txt<bias59_t, 512>(b59_0, "b59_0.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer97_t> layer97_out[N_CHAN_97/256];
    #pragma HLS STREAM variable=layer97_out depth=81
    nnet::zeropad2d_cl<layer56_t, layer97_t, config97>(layer56_out, layer97_out); // zp2d_q_conv2d_batchnorm_16


    hls::stream<layer73_t> layer73_out1[N_FILT_73/256];
    #pragma HLS STREAM variable=layer73_out1 depth=81
    #pragma HLS BIND_STORAGE variable=layer73_out1 type=fifo impl=srl
    hls::stream<layer73_t> layer73_out2[N_FILT_73/256];
    #pragma HLS STREAM variable=layer73_out2 depth=81
    #pragma HLS BIND_STORAGE variable=layer73_out2 type=fifo impl=srl
    nnet::split2<layer97_t, layer73_t, config59_split>(layer97_out, layer73_out1, layer73_out2);   // 为了得到数组格式的流数据，不然直接两个流分别运算了


    hls::stream<layer62_t> layer62_out1[N_FILT_59/256];
    #pragma HLS STREAM variable=layer62_out1 depth=49
    #pragma HLS BIND_STORAGE variable=layer62_out1 type=fifo impl=srl
    nnet::conv_2d_cl<layer73_t, layer62_t, config59_1>(layer73_out1, layer62_out1, w59_1_1, b59); // q_conv2d_batchnorm_16_1


    hls::stream<layer62_t> layer62_out2[N_FILT_59/256];
    #pragma HLS STREAM variable=layer62_out2 depth=49
    nnet::conv_2d_cl<layer73_t, layer62_t, config59_2>(layer73_out2, layer62_out2, w59_2_1, b59_0); // q_conv2d_batchnorm_16_2

    nnet::add2<layer62_t, config59_add>(layer62_out1, layer62_out2, layer62_out);
}
