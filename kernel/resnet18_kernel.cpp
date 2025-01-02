#include <iostream>
#include <string>

#include "resnet18_split.h"
#include "resnet18_kernel.h"



// hls-fpga-machine-learning start prepare
void prepare_data_kernel(galapagos_interface *out_1) { 
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS INTERFACE axis register both port=out_1 

  //insert input data source
    input_t input_val[N_CASE][N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1] = { 
        #include "../tb_data/tb_input_features.dat"
    };

    //insert input data define
    hls::stream<input_t> input_1[N_INPUT_3_1/3];
    #pragma HLS STREAM variable=input_1 depth=50176

    for (int i=0;i < N_CASE;i++) {
        // insert input data funciton send define
        nnet::copy_data<input_t, input_t, 0, N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1, N_INPUT_3_1>(input_val[i], input_1);
        nnet::hls_stream_2_galapagos_interface<input_t, N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1/N_INPUT_3_1, 8, N_INPUT_3_1>(input_1, out_1, 1, 1);
    }
}


// hls-fpga-machine-learning start receive
void recv_data_kernel(galapagos_interface *in_0)  { 
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis register both port=in_0

    //insert output data define
    hls::stream<result_t> layer74_out[N_LAYER_72/200];
    #pragma HLS STREAM variable=layer74_out depth=1

    for (int i=0;i < N_CASE;i++) {
        // insert output data funciton receive define
        nnet::galapagos_interface_2_hls_stream<result_t, N_LAYER_72/N_LAYER_72, 8, N_LAYER_72>(in_0, layer74_out);

    #ifdef RECV_SIM  //仿真输出结果用
        // insert output data print
        nnet::print_result<result_t,N_LAYER_72, N_LAYER_72>(layer74_out,std::cout);
    #endif
    }
}


// hls-fpga-machine-learning start send node
void send(galapagos_interface *in_0, galapagos_interface *out_1) {
#pragma HLS INTERFACE ap_ctrl_none port=return   
#pragma HLS INTERFACE axis register both port=in_0
#pragma HLS INTERFACE axis register both port=out_1  
  
    //insert input data source
    input_t input_val[N_CASE][N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1] = { 
        #include "../tb_data/tb_input_features.dat"
    };

    //insert input data define
    hls::stream<input_t> input_1[N_INPUT_3_1/3];
    #pragma HLS STREAM variable=input_1 depth=50176

    for (int i=0;i < N_CASE;i++) {
        // insert input data funciton send define
        nnet::copy_data<input_t, input_t, 0, N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1, N_INPUT_3_1>(input_val[i], input_1);
        nnet::hls_stream_2_galapagos_interface<input_t, N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1/N_INPUT_3_1, 8, N_INPUT_3_1>(input_1, out_1, 1, 1);
    }

    for (int i=0;i < N_CASE;i++) {
        // insert output data funciton receive define
        nnet::read_galapagos_interface_data<result_t, N_LAYER_72/N_LAYER_72, 8, N_LAYER_72>(in_0);
    }
}


// hls-fpga-machine-learning start compute
void kernel1_wrapper(galapagos_interface *in_1, galapagos_interface *out_2){
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis register both port=in_1
#pragma HLS INTERFACE axis register both port=out_2

    // hls-fpga-machine-learning insert variables define
    hls::stream<input_t> input_1[N_INPUT_3_1/3];
    #pragma HLS STREAM variable=input_1 depth=50176
    //#pragma HLS BIND_STORAGE variable=input_1 type=fifo impl=srl    //size of variable 'SRL_SIG' is too large to handle; the size of the variable is 1204224, the limit is 1000000.
    hls::stream<layer12_t> layer12_out[N_FILT_5/64];
    #pragma HLS STREAM variable=layer12_out depth=3136
    //#pragma HLS BIND_STORAGE variable=layer12_out type=fifo impl=srl    //size of variable 'SRL_SIG' is too large to handle; the size of the variable is 1605632, the limit is 1000000.
    
    for (int i=0;i < N_CASE;i++) {
        // hls-fpga-machine-learning insert functions in define
        nnet::galapagos_interface_2_hls_stream<input_t, N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1/N_INPUT_3_1, 8, N_INPUT_3_1>(in_1, input_1);

        // hls-fpga-machine-learning insert functions kernel define
        kernel1(input_1,layer12_out);

        // hls-fpga-machine-learning insert functions out define
        nnet::hls_stream_2_galapagos_interface<layer12_t, OUT_HEIGHT_5*OUT_WIDTH_5*N_FILT_5/N_FILT_5, 8, N_FILT_5>(layer12_out, out_2, 2, 2);
    }

}

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

