#include <iostream>
#include <string>

#include "resnet18_split2.h"
#include "resnet18_kernel2.h"


// hls-fpga-machine-learning start compute
void kernel2_wrapper(galapagos_interface *in_2, galapagos_interface *out_3){
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis register both port=in_2
#pragma HLS INTERFACE axis register both port=out_3

    // hls-fpga-machine-learning insert variables define
    hls::stream<layer12_t> layer12_out[N_FILT_5/64];
    #pragma HLS STREAM variable=layer12_out depth=3136
    //#pragma HLS BIND_STORAGE variable=layer12_out type=fifo impl=srl    //size of variable 'SRL_SIG' is too large to handle; the size of the variable is 1605632, the limit is 1000000.
    hls::stream<layer29_t> layer29_out[N_FILT_23/128];
    #pragma HLS STREAM variable=layer29_out depth=784
    //#pragma HLS BIND_STORAGE variable=layer29_out type=fifo impl=srl
    
    for (int i=0;i < N_CASE;i++) {
        // hls-fpga-machine-learning insert functions in define
        nnet::galapagos_interface_2_hls_stream<layer12_t, OUT_HEIGHT_5*OUT_WIDTH_5*N_FILT_5/N_FILT_5, 8, N_FILT_5>(in_2, layer12_out);

        // hls-fpga-machine-learning insert functions kernel define
        kernel2(layer12_out,layer29_out);

        // hls-fpga-machine-learning insert functions out define
        nnet::hls_stream_2_galapagos_interface<layer29_t, OUT_HEIGHT_23*OUT_WIDTH_23*N_FILT_23/N_FILT_23, 8, N_FILT_23>(layer29_out, out_3, 3, 3);
    }

}

