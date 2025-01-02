#include <iostream>
#include <string>

#include "resnet18_split1.h"
#include "resnet18_kernel1.h"


// hls-fpga-machine-learning start compute
void kernel1_wrapper(galapagos_interface *in_1, galapagos_interface *out_2){
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis register both port=in_1
#pragma HLS INTERFACE axis register both port=out_2

    // hls-fpga-machine-learning insert variables define
    hls::stream<input_t> input_1[N_INPUT_3_1/3];
    #pragma HLS STREAM variable=input_1 depth=50176
    //#pragma HLS BIND_STORAGE variable=input_1 type=fifo impl=srl    //size of variable 'SRL_SIG' is too large to handle; the size of the variable is 1204224, the limit is 1000000.
    hls::stream<layer12_t> layer12_out[N_FILT_5/64];
    #pragma HLS STREAM variable=layer12_out depth=3136
    //#pragma HLS BIND_STORAGE variable=layer12_out type=fifo impl=srl    //size of variable 'SRL_SIG' is too large to handle; the size of the variable is 1605632, the limit is 1000000.
    
    for (int i=0;i < N_CASE;i++) {
        // hls-fpga-machine-learning insert functions in define
        nnet::galapagos_interface_2_hls_stream<input_t, N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1/N_INPUT_3_1, 8, N_INPUT_3_1>(in_1, input_1);

        // hls-fpga-machine-learning insert functions kernel define
        kernel1(input_1,layer12_out);

        // hls-fpga-machine-learning insert functions out define
        nnet::hls_stream_2_galapagos_interface<layer12_t, OUT_HEIGHT_5*OUT_WIDTH_5*N_FILT_5/N_FILT_5, 8, N_FILT_5>(layer12_out, out_2, 2, 2);
    }

}

