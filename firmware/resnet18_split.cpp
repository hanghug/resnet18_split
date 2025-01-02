#include <iostream>

#include "resnet18_split.h"
#include "parameters.h"

// hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/w6.h"
#include "weights/b6.h"
#include "weights/w9.h"
#include "weights/b9.h"
#include "weights/w13.h"
#include "weights/b13.h"
#include "weights/w16.h"
#include "weights/b16.h"
#include "weights/w20.h"
#include "weights/b20.h"
#include "weights/w100.h"
#include "weights/b100.h"
#include "weights/w25.h"
#include "weights/b25.h"
#include "weights/w30.h"
#include "weights/b30.h"
#include "weights/w33.h"
#include "weights/b33.h"
#include "weights/w37.h"
#include "weights/b37.h"
#include "weights/w101.h"
#include "weights/b101.h"
#include "weights/w42.h"
#include "weights/b42.h"
#include "weights/w47.h"
#include "weights/b47.h"
#include "weights/w50.h"
#include "weights/b50.h"
#include "weights/w54.h"
#include "weights/b54.h"
#include "weights/w102.h"
#include "weights/b102.h"
#include "weights/w59_1_1.h"
#include "weights/w59_2_1.h"
#include "weights/b59.h"
#include "weights/b59_0.h"
#include "weights/w64_1_1.h"
#include "weights/w64_2_1.h"
#include "weights/b64.h"
#include "weights/b64_0.h"
#include "weights/w67_1_1.h"
#include "weights/w67_2_1.h"
#include "weights/b67.h"
#include "weights/b67_0.h"
#include "weights/w72.h"
#include "weights/b72.h"

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



void kernel8(
    hls::stream<layer53_t> (&layer81_cpy2)[N_FILT_40/256],
    hls::stream<layer61_t> (&layer61_out)[N_FILT_57/256]
) {
    // hls-fpga-machine-learning insert IO
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight102_t, 131072>(w102, "w102.txt");
        nnet::load_weights_from_txt<bias102_t, 512>(b102, "b102.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    nnet::pointwise_conv_2d_cl<layer53_t, layer102_t, config102>(layer81_cpy2, layer61_out, w102, b102); // q_conv2d_batchnorm_17

}


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
    //#pragma HLS BIND_STORAGE variable=layer71_out type=fifo impl=srl
    nnet::global_pooling2d_cl<layer70_t, layer71_t, config71>(layer70_out, layer71_out); // global_average_pooling2d

    hls::stream<layer72_t> layer72_out[N_LAYER_72/200];
    #pragma HLS STREAM variable=layer72_out depth=1
    //#pragma HLS BIND_STORAGE variable=layer72_out type=fifo impl=srl
    nnet::dense<layer71_t, layer72_t, config72>(layer71_out, layer72_out, w72, b72); // q_dense

    nnet::softmax<layer72_t, result_t, softmax_config74>(layer72_out, layer74_out); // activation_20

}


