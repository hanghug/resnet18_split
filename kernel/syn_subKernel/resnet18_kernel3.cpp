#include <iostream>
#include <string>

#include "resnet18_split3.h"
#include "resnet18_kernel3.h"


// hls-fpga-machine-learning start compute
void kernel3_wrapper(galapagos_interface *in_3, galapagos_interface *out_4){
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis register both port=in_3
#pragma HLS INTERFACE axis register both port=out_4

    // hls-fpga-machine-learning insert variables define
    hls::stream<layer29_t> layer29_out[N_FILT_23/128];
    #pragma HLS STREAM variable=layer29_out depth=784
    //#pragma HLS BIND_STORAGE variable=layer29_out type=fifo impl=srl
    hls::stream<layer36_t> layer36_out[N_FILT_23/128];
    #pragma HLS STREAM variable=layer36_out depth=784
    //#pragma HLS BIND_STORAGE variable=layer36_out type=fifo impl=srl
    
    for (int i=0;i < N_CASE;i++) {
        // hls-fpga-machine-learning insert functions in define
        nnet::galapagos_interface_2_hls_stream<layer29_t, OUT_HEIGHT_23*OUT_WIDTH_23*N_FILT_23/N_FILT_23, 8, N_FILT_23>(in_3, layer29_out);

        // hls-fpga-machine-learning insert functions kernel define
        kernel3(layer29_out,layer36_out);

        // hls-fpga-machine-learning insert functions out define
        nnet::hls_stream_2_galapagos_interface<layer36_t, OUT_HEIGHT_23*OUT_WIDTH_23*N_FILT_23/N_FILT_23, 8, N_FILT_23>(layer36_out, out_4, 4, 4);
    }

}

