#include <iostream>
#include <string>

#include "resnet18_split11.h"
#include "resnet18_kernel11.h"


// hls-fpga-machine-learning start compute
void kernel11_wrapper(galapagos_interface *in_13, galapagos_interface *out_0){
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis register both port=in_13
#pragma HLS INTERFACE axis register both port=out_0

    // hls-fpga-machine-learning insert variables define
    hls::stream<layer70_t> layer70_out[N_FILT_57/256];
    #pragma HLS STREAM variable=layer70_out depth=49
    //#pragma HLS BIND_STORAGE variable=layer70_out type=fifo impl=srl
    hls::stream<result_t> layer74_out[N_LAYER_72/200];
    #pragma HLS STREAM variable=layer74_out depth=1
    //#pragma HLS BIND_STORAGE variable=layer74_out type=fifo impl=srl
    
    for (int i=0;i < N_CASE;i++) {
        // hls-fpga-machine-learning insert functions in define
        nnet::galapagos_interface_2_hls_stream<layer70_t, OUT_HEIGHT_57*OUT_WIDTH_57*N_FILT_57/N_FILT_57, 8, N_FILT_57>(in_13, layer70_out);

        // hls-fpga-machine-learning insert functions kernel define
        kernel11(layer70_out,layer74_out);

        // hls-fpga-machine-learning insert functions out define
        nnet::hls_stream_2_galapagos_interface<result_t, N_LAYER_72/N_LAYER_72, 8, N_LAYER_72>(layer74_out, out_0, 0, 0);
    }

}

