#include <iostream>
#include <string>

#include "resnet18_split8.h"
#include "resnet18_kernel8.h"


// hls-fpga-machine-learning start compute
void kernel8_wrapper(galapagos_interface *in_8, galapagos_interface *out_9){
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis register both port=in_8
#pragma HLS INTERFACE axis register both port=out_9

    // hls-fpga-machine-learning insert variables define
    hls::stream<layer53_t> layer81_cpy2[N_FILT_40/256];
    #pragma HLS STREAM variable=layer81_cpy2 depth=196
    //#pragma HLS BIND_STORAGE variable=layer81_cpy2 type=fifo impl=srl
    hls::stream<layer61_t> layer61_out[N_FILT_57/256];
    #pragma HLS STREAM variable=layer61_out depth=49
    //#pragma HLS BIND_STORAGE variable=layer61_out type=fifo impl=srl
    
    for (int i=0;i < N_CASE;i++) {
        // hls-fpga-machine-learning insert functions in define
        nnet::galapagos_interface_2_hls_stream<layer53_t, OUT_HEIGHT_40*OUT_WIDTH_40*N_FILT_40/N_FILT_40, 8, N_FILT_40>(in_8, layer81_cpy2);

        // hls-fpga-machine-learning insert functions kernel define
        kernel8(layer81_cpy2,layer61_out);

        // hls-fpga-machine-learning insert functions out define
        nnet::hls_stream_2_galapagos_interface<layer61_t, OUT_HEIGHT_57*OUT_WIDTH_57*N_FILT_57/N_FILT_57, 8, N_FILT_57>(layer61_out, out_9, 9, 9);
    }

}

