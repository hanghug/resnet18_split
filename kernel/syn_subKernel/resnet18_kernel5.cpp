#include <iostream>
#include <string>

#include "resnet18_split5.h"
#include "resnet18_kernel5.h"


// hls-fpga-machine-learning start compute
void kernel5_wrapper(galapagos_interface *in_5, galapagos_interface *out_6){
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis register both port=in_5
#pragma HLS INTERFACE axis register both port=out_6

    // hls-fpga-machine-learning insert variables define
    hls::stream<layer46_t> layer46_out[N_FILT_40/256];
    #pragma HLS STREAM variable=layer46_out depth=196
    //#pragma HLS BIND_STORAGE variable=layer46_out type=fifo impl=srl
    hls::stream<layer53_t> layer53_out[N_FILT_40/256];
    #pragma HLS STREAM variable=layer53_out depth=196
    //#pragma HLS BIND_STORAGE variable=layer53_out type=fifo impl=srl
    
    for (int i=0;i < N_CASE;i++) {
        // hls-fpga-machine-learning insert functions in define
        nnet::galapagos_interface_2_hls_stream<layer46_t, OUT_HEIGHT_40*OUT_WIDTH_40*N_FILT_40/N_FILT_40, 8, N_FILT_40>(in_5, layer46_out);

        // hls-fpga-machine-learning insert functions kernel define
        kernel5(layer46_out,layer53_out);

        // hls-fpga-machine-learning insert functions out define
        nnet::hls_stream_2_galapagos_interface<layer53_t, OUT_HEIGHT_40*OUT_WIDTH_40*N_FILT_40/N_FILT_40, 8, N_FILT_40>(layer53_out, out_6, 6, 6);
    }

}

