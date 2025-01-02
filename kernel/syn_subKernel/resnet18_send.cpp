#include <iostream>
#include <string>

#include "resnet18_send.h"
#include "defines.h"


// hls-fpga-machine-learning start send node
void send(galapagos_interface *in_0, galapagos_interface *out_1) {
#pragma HLS INTERFACE ap_ctrl_none port=return   
#pragma HLS INTERFACE axis register both port=in_0
#pragma HLS INTERFACE axis register both port=out_1  
  
    input_t input_val[N_CASE][N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1] = { 
        #include "../tb_data/tb_input_features.dat"
    };

    hls::stream<input_t> input_1[N_INPUT_3_1/3];
    #pragma HLS STREAM variable=input_1 depth=50176

    for (int i=0;i < N_CASE;i++) {
        nnet::copy_data<input_t, input_t, 0, N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1, N_INPUT_3_1>(input_val[i], input_1);
        nnet::hls_stream_2_galapagos_interface<input_t, N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1/N_INPUT_3_1, 8, N_INPUT_3_1>(input_1, out_1, 1, 1);
    }

    for (int i=0;i < N_CASE;i++) {
        nnet::read_galapagos_interface_data<result_t, N_LAYER_72/N_LAYER_72, 8, N_LAYER_72>(in_0);
    }
}



