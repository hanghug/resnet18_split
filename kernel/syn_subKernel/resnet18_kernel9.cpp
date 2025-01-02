#include <iostream>
#include <string>

#include "resnet18_split9.h"
#include "resnet18_kernel9.h"


// hls-fpga-machine-learning start compute
void kernel9_wrapper(galapagos_interface *in_9, galapagos_interface *in_10, galapagos_interface *out_11, galapagos_interface *out_12){
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis register both port=in_9
#pragma HLS INTERFACE axis register both port=in_10
#pragma HLS INTERFACE axis register both port=out_11
#pragma HLS INTERFACE axis register both port=out_12

    // hls-fpga-machine-learning insert variables define
    hls::stream<layer61_t> layer61_out[N_FILT_57/256];
    #pragma HLS STREAM variable=layer61_out depth=49
    //#pragma HLS BIND_STORAGE variable=layer61_out type=fifo impl=srl
    hls::stream<layer62_t> layer62_out[N_FILT_59/256];
    #pragma HLS STREAM variable=layer62_out depth=49
    //#pragma HLS BIND_STORAGE variable=layer62_out type=fifo impl=srl
    hls::stream<layer66_t> layer66_out[N_FILT_64/256];
    #pragma HLS STREAM variable=layer66_out depth=49
    //#pragma HLS BIND_STORAGE variable=layer66_out type=fifo impl=srl
    hls::stream<layer63_t> layer82_cpy2[N_FILT_57/256];
    #pragma HLS STREAM variable=layer82_cpy2 depth=49
    //#pragma HLS BIND_STORAGE variable=layer82_cpy2 type=fifo impl=srl
    
    for (int i=0;i < N_CASE;i++) {
        // hls-fpga-machine-learning insert functions in define
        nnet::galapagos_interface_2_hls_stream<layer61_t, OUT_HEIGHT_57*OUT_WIDTH_57*N_FILT_57/N_FILT_57, 8, N_FILT_57>(in_9, layer61_out);
        nnet::galapagos_interface_2_hls_stream<layer62_t, OUT_HEIGHT_59*OUT_WIDTH_59*N_FILT_59/N_FILT_59, 8, N_FILT_59>(in_10, layer62_out);

        // hls-fpga-machine-learning insert functions kernel define
        kernel9(layer61_out,layer62_out,layer66_out,layer82_cpy2);

        // hls-fpga-machine-learning insert functions out define
        nnet::hls_stream_2_galapagos_interface<layer66_t, OUT_HEIGHT_64*OUT_WIDTH_64*N_FILT_64/N_FILT_64, 8, N_FILT_64>(layer66_out, out_11, 11, 11);
        nnet::hls_stream_2_galapagos_interface<layer63_t, OUT_HEIGHT_57*OUT_WIDTH_57*N_FILT_57/N_FILT_57, 8, N_FILT_57>(layer82_cpy2, out_12, 12, 12);
    }

}

