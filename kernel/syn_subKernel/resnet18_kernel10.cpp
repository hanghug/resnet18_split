#include <iostream>
#include <string>

#include "resnet18_split10.h"
#include "resnet18_kernel10.h"


// hls-fpga-machine-learning start compute
void kernel10_wrapper(galapagos_interface *in_11, galapagos_interface *in_12, galapagos_interface *out_13){
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis register both port=in_11
#pragma HLS INTERFACE axis register both port=in_12
#pragma HLS INTERFACE axis register both port=out_13

    // hls-fpga-machine-learning insert variables define
    hls::stream<layer66_t> layer66_out[N_FILT_64/256];
    #pragma HLS STREAM variable=layer66_out depth=49
    //#pragma HLS BIND_STORAGE variable=layer66_out type=fifo impl=srl
    hls::stream<layer63_t> layer82_cpy2[N_FILT_57/256];
    #pragma HLS STREAM variable=layer82_cpy2 depth=49
    //#pragma HLS BIND_STORAGE variable=layer82_cpy2 type=fifo impl=srl
    hls::stream<layer70_t> layer70_out[N_FILT_57/256];
    #pragma HLS STREAM variable=layer70_out depth=49
    //#pragma HLS BIND_STORAGE variable=layer70_out type=fifo impl=srl
    
    for (int i=0;i < N_CASE;i++) {
        // hls-fpga-machine-learning insert functions in define
        nnet::galapagos_interface_2_hls_stream<layer66_t, OUT_HEIGHT_64*OUT_WIDTH_64*N_FILT_64/N_FILT_64, 8, N_FILT_64>(in_11, layer66_out);
        nnet::galapagos_interface_2_hls_stream<layer63_t, OUT_HEIGHT_57*OUT_WIDTH_57*N_FILT_57/N_FILT_57, 8, N_FILT_57>(in_12, layer82_cpy2);

        // hls-fpga-machine-learning insert functions kernel define
        kernel10(layer66_out,layer82_cpy2,layer70_out);

        // hls-fpga-machine-learning insert functions out define
        nnet::hls_stream_2_galapagos_interface<layer70_t, OUT_HEIGHT_57*OUT_WIDTH_57*N_FILT_57/N_FILT_57, 8, N_FILT_57>(layer70_out, out_13, 13, 13);
    }

}

