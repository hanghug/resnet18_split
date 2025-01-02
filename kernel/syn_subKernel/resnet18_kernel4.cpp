#include <iostream>
#include <string>

#include "resnet18_split4.h"
#include "resnet18_kernel4.h"


// hls-fpga-machine-learning start compute
void kernel4_wrapper(galapagos_interface *in_4, galapagos_interface *out_5){
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis register both port=in_4
#pragma HLS INTERFACE axis register both port=out_5

    // hls-fpga-machine-learning insert variables define
    hls::stream<layer36_t> layer36_out[N_FILT_23/128];
    #pragma HLS STREAM variable=layer36_out depth=784
    //#pragma HLS BIND_STORAGE variable=layer36_out type=fifo impl=srl
    hls::stream<layer46_t> layer46_out[N_FILT_40/256];
    #pragma HLS STREAM variable=layer46_out depth=196
    //#pragma HLS BIND_STORAGE variable=layer46_out type=fifo impl=srl
    
    for (int i=0;i < N_CASE;i++) {
        // hls-fpga-machine-learning insert functions in define
        nnet::galapagos_interface_2_hls_stream<layer36_t, OUT_HEIGHT_23*OUT_WIDTH_23*N_FILT_23/N_FILT_23, 8, N_FILT_23>(in_4, layer36_out);

        // hls-fpga-machine-learning insert functions kernel define
        kernel4(layer36_out,layer46_out);

        // hls-fpga-machine-learning insert functions out define
        nnet::hls_stream_2_galapagos_interface<layer46_t, OUT_HEIGHT_40*OUT_WIDTH_40*N_FILT_40/N_FILT_40, 8, N_FILT_40>(layer46_out, out_5, 5, 5);
    }

}

