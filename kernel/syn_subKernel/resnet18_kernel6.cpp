#include <iostream>
#include <string>

#include "resnet18_split6.h"
#include "resnet18_kernel6.h"


// hls-fpga-machine-learning start compute
void kernel6_wrapper(galapagos_interface *in_6, galapagos_interface *out_7, galapagos_interface *out_8){
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis register both port=in_6
#pragma HLS INTERFACE axis register both port=out_7
#pragma HLS INTERFACE axis register both port=out_8

    // hls-fpga-machine-learning insert variables define
    hls::stream<layer53_t> layer53_out[N_FILT_40/256];
    #pragma HLS STREAM variable=layer53_out depth=196
    //#pragma HLS BIND_STORAGE variable=layer53_out type=fifo impl=srl
    hls::stream<layer56_t> layer56_out[N_FILT_54/256];
    #pragma HLS STREAM variable=layer56_out depth=49
    //#pragma HLS BIND_STORAGE variable=layer56_out type=fifo impl=srl
    hls::stream<layer53_t> layer81_cpy2[N_FILT_40/256];
    #pragma HLS STREAM variable=layer81_cpy2 depth=196
    //#pragma HLS BIND_STORAGE variable=layer81_cpy2 type=fifo impl=srl
    
    for (int i=0;i < N_CASE;i++) {
        // hls-fpga-machine-learning insert functions in define
        nnet::galapagos_interface_2_hls_stream<layer53_t, OUT_HEIGHT_40*OUT_WIDTH_40*N_FILT_40/N_FILT_40, 8, N_FILT_40>(in_6, layer53_out);

        // hls-fpga-machine-learning insert functions kernel define
        kernel6(layer53_out,layer56_out,layer81_cpy2);

        // hls-fpga-machine-learning insert functions out define
        nnet::hls_stream_2_galapagos_interface<layer56_t, OUT_HEIGHT_54*OUT_WIDTH_54*N_FILT_54/N_FILT_54, 8, N_FILT_54>(layer56_out, out_7, 7, 7);
        nnet::hls_stream_2_galapagos_interface<layer53_t, OUT_HEIGHT_40*OUT_WIDTH_40*N_FILT_40/N_FILT_40, 8, N_FILT_40>(layer81_cpy2, out_8, 8, 8);
    }

}

