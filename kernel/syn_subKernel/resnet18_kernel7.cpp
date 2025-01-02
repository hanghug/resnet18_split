#include <iostream>
#include <string>

#include "resnet18_split7.h"
#include "resnet18_kernel7.h"


// hls-fpga-machine-learning start compute
void kernel7_wrapper(galapagos_interface *in_7, galapagos_interface *out_10){
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis register both port=in_7
#pragma HLS INTERFACE axis register both port=out_10

    // hls-fpga-machine-learning insert variables define
    hls::stream<layer56_t> layer56_out[N_FILT_54/256];
    #pragma HLS STREAM variable=layer56_out depth=49
    //#pragma HLS BIND_STORAGE variable=layer56_out type=fifo impl=srl
    hls::stream<layer62_t> layer62_out[N_FILT_59/256];
    #pragma HLS STREAM variable=layer62_out depth=49
    //#pragma HLS BIND_STORAGE variable=layer62_out type=fifo impl=srl
    
    for (int i=0;i < N_CASE;i++) {
        // hls-fpga-machine-learning insert functions in define
        nnet::galapagos_interface_2_hls_stream<layer56_t, OUT_HEIGHT_54*OUT_WIDTH_54*N_FILT_54/N_FILT_54, 8, N_FILT_54>(in_7, layer56_out);

        // hls-fpga-machine-learning insert functions kernel define
        kernel7(layer56_out,layer62_out);

        // hls-fpga-machine-learning insert functions out define
        nnet::hls_stream_2_galapagos_interface<layer62_t, OUT_HEIGHT_59*OUT_WIDTH_59*N_FILT_59/N_FILT_59, 8, N_FILT_59>(layer62_out, out_10, 10, 10);
    }

}

