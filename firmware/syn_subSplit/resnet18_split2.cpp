#include <iostream>

#include "resnet18_split2.h"
#include "parameters.h"

// hls-fpga-machine-learning insert weights
#include "weights/w13.h"
#include "weights/b13.h"
#include "weights/w16.h"
#include "weights/b16.h"
#include "weights/w20.h"
#include "weights/b20.h"
#include "weights/w25.h"
#include "weights/b25.h"
#include "weights/w100.h"
#include "weights/b100.h"

// hls-fpga-machine-learning start
void kernel2(
    hls::stream<layer12_t> (&layer12_out)[N_FILT_5/64],
    hls::stream<layer29_t> (&layer29_out)[N_FILT_23/128]
) {
    // hls-fpga-machine-learning insert IO
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight13_t, 36864>(w13, "w13.txt");
        nnet::load_weights_from_txt<bias13_t, 64>(b13, "b13.txt");
        nnet::load_weights_from_txt<weight16_t, 36864>(w16, "w16.txt");
        nnet::load_weights_from_txt<bias16_t, 64>(b16, "b16.txt");
        nnet::load_weights_from_txt<weight20_t, 73728>(w20, "w20.txt");
        nnet::load_weights_from_txt<bias20_t, 128>(b20, "b20.txt");
        nnet::load_weights_from_txt<weight25_t, 147456>(w25, "w25.txt");
        nnet::load_weights_from_txt<bias25_t, 128>(b25, "b25.txt");
        nnet::load_weights_from_txt<weight100_t, 8192>(w100, "w100.txt");
        nnet::load_weights_from_txt<bias100_t, 128>(b100, "b100.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer12_t> layer76_cpy1[N_FILT_5/64];
    #pragma HLS STREAM variable=layer76_cpy1 depth=1024
    //#pragma HLS BIND_STORAGE variable=layer76_cpy1 type=fifo impl=srl    //size of variable 'SRL_SIG' is too large to handle; the size of the variable is 1605632, the limit is 1000000.
    hls::stream<layer12_t> layer76_cpy2[N_FILT_5/64];
    #pragma HLS STREAM variable=layer76_cpy2 depth=1024
    //#pragma HLS BIND_STORAGE variable=layer76_cpy2 type=fifo impl=srl    //size of variable 'SRL_SIG' is too large to handle; the size of the variable is 1605632, the limit is 1000000.
    nnet::clone_stream<layer12_t, layer12_t, 200704, N_FILT_5>(layer12_out, layer76_cpy1, layer76_cpy2); // clone_add

    hls::stream<layer86_t> layer86_out[N_CHAN_86/64];
    #pragma HLS STREAM variable=layer86_out depth=1024
    //#pragma HLS BIND_STORAGE variable=layer86_out type=fifo impl=srl    //size of variable 'SRL_SIG' is too large to handle; the size of the variable is 1722368, the limit is 1000000.
    nnet::zeropad2d_cl<layer12_t, layer86_t, config86>(layer76_cpy1, layer86_out); // zp2d_q_conv2d_batchnorm_3

    hls::stream<layer13_t> layer13_out[N_FILT_13/64];
    #pragma HLS STREAM variable=layer13_out depth=1024
    //#pragma HLS BIND_STORAGE variable=layer13_out type=fifo impl=srl    //size of variable 'SRL_SIG' is too large to handle; the size of the variable is 1605632, the limit is 1000000.
    nnet::conv_2d_cl<layer86_t, layer13_t, config13>(layer86_out, layer13_out, w13, b13); // q_conv2d_batchnorm_3

    hls::stream<layer87_t> layer87_out[N_CHAN_87/64];
    #pragma HLS STREAM variable=layer87_out depth=3364
    //#pragma HLS BIND_STORAGE variable=layer87_out type=fifo impl=srl    //size of variable 'SRL_SIG' is too large to handle; the size of the variable is 1722368, the limit is 1000000.
    nnet::zeropad2d_cl<layer15_t, layer87_t, config87>(layer13_out, layer87_out); // zp2d_q_conv2d_batchnorm_4

    hls::stream<layer16_t> layer16_out[N_FILT_16/64];
    #pragma HLS STREAM variable=layer16_out depth=1024
    //#pragma HLS BIND_STORAGE variable=layer16_out type=fifo impl=srl    //size of variable 'SRL_SIG' is too large to handle; the size of the variable is 1605632, the limit is 1000000.
    nnet::conv_2d_cl<layer87_t, layer16_t, config16>(layer87_out, layer16_out, w16, b16); // q_conv2d_batchnorm_4

    hls::stream<layer19_t> layer19_out[N_FILT_5/64];
    #pragma HLS STREAM variable=layer19_out depth=1024
    //#pragma HLS BIND_STORAGE variable=layer19_out type=fifo impl=srl    //size of variable 'SRL_SIG' is too large to handle; the size of the variable is 1605632, the limit is 1000000.
    nnet::add<layer12_t, layer18_t, layer19_t, config19, N_FILT_5>(layer76_cpy2, layer16_out, layer19_out); // add_1

    hls::stream<layer19_t> layer77_cpy1[N_FILT_5/64];
    #pragma HLS STREAM variable=layer77_cpy1 depth=1024
    //#pragma HLS BIND_STORAGE variable=layer77_cpy1 type=fifo impl=srl    //size of variable 'SRL_SIG' is too large to handle; the size of the variable is 1605632, the limit is 1000000.
    hls::stream<layer19_t> layer77_cpy2[N_FILT_5/64];
    #pragma HLS STREAM variable=layer77_cpy2 depth=1024
    //#pragma HLS BIND_STORAGE variable=layer77_cpy2 type=fifo impl=srl    //size of variable 'SRL_SIG' is too large to handle; the size of the variable is 1605632, the limit is 1000000.
    nnet::clone_stream<layer19_t, layer19_t, 200704, N_FILT_5>(layer19_out, layer77_cpy1, layer77_cpy2); // clone_add_1

    hls::stream<layer88_t> layer88_out[N_CHAN_88/64];
    #pragma HLS STREAM variable=layer88_out depth=1024
    //#pragma HLS BIND_STORAGE variable=layer88_out type=fifo impl=srl    //size of variable 'SRL_SIG' is too large to handle; the size of the variable is 1663488, the limit is 1000000.
    nnet::zeropad2d_cl<layer19_t, layer88_t, config88>(layer77_cpy1, layer88_out); // zp2d_q_conv2d_batchnorm_5

    hls::stream<layer20_t> layer20_out[N_FILT_20/128];
    #pragma HLS STREAM variable=layer20_out depth=784
    //#pragma HLS BIND_STORAGE variable=layer20_out type=fifo impl=srl
    nnet::conv_2d_cl<layer88_t, layer20_t, config20>(layer88_out, layer20_out, w20, b20); // q_conv2d_batchnorm_5

    hls::stream<layer89_t> layer89_out[N_CHAN_89/128];
    #pragma HLS STREAM variable=layer89_out depth=900
    //#pragma HLS BIND_STORAGE variable=layer89_out type=fifo impl=srl
    nnet::zeropad2d_cl<layer22_t, layer89_t, config89>(layer20_out, layer89_out); // zp2d_q_conv2d_batchnorm_6

    hls::stream<layer25_t> layer25_out[N_FILT_25/128];
    #pragma HLS STREAM variable=layer25_out depth=784
    //#pragma HLS BIND_STORAGE variable=layer25_out type=fifo impl=srl
    nnet::conv_2d_cl<layer89_t, layer25_t, config25>(layer89_out, layer25_out, w25, b25); // q_conv2d_batchnorm_6

    hls::stream<layer100_t> layer100_out[N_FILT_100/128];
    #pragma HLS STREAM variable=layer100_out depth=784
    //#pragma HLS BIND_STORAGE variable=layer100_out type=fifo impl=srl
    nnet::pointwise_conv_2d_cl<layer19_t, layer100_t, config100>(layer77_cpy2, layer100_out, w100, b100); // q_conv2d_batchnorm_7

    nnet::add<layer27_t, layer28_t, layer29_t, config29, N_FILT_23>(layer100_out, layer25_out, layer29_out); // add_2

}