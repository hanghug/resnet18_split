#ifndef RESNET18_KERNEL_H_
#define RESNET18_KERNEL_H_

// #define CPU

#ifdef CPU
#include "galapagos_interface.hpp"
#include "nnet_utils/nnet_spdlog.h" 
#else
#include "galapagos_packet.h"
#endif

#include "nnet_utils/nnet_helpers.h"
#include "nnet_utils/nnet_types.h"
#include "nnet_utils/nnet_stream2gala.h"


#define N_CASE 1 

void prepare_data_kernel(galapagos_interface *out_1);
void recv_data_kernel(galapagos_interface *in_0);
void send(galapagos_interface *in_0, galapagos_interface *out_1);

// hls-fpga-machine-learning insert functions define
void kernel1_wrapper(galapagos_interface *in_1, galapagos_interface *out_2);
void kernel2_wrapper(galapagos_interface *in_2, galapagos_interface *out_3);
void kernel3_wrapper(galapagos_interface *in_3, galapagos_interface *out_4);
void kernel4_wrapper(galapagos_interface *in_4, galapagos_interface *out_5);
void kernel5_wrapper(galapagos_interface *in_5, galapagos_interface *out_6);
void kernel6_wrapper(galapagos_interface *in_6, galapagos_interface *out_7, galapagos_interface *out_8);
void kernel7_wrapper(galapagos_interface *in_7, galapagos_interface *out_10);
void kernel8_wrapper(galapagos_interface *in_8, galapagos_interface *out_9);
void kernel9_wrapper(galapagos_interface *in_9, galapagos_interface *in_10, galapagos_interface *out_11, galapagos_interface *out_12);
void kernel10_wrapper(galapagos_interface *in_11, galapagos_interface *in_12, galapagos_interface *out_13);
void kernel11_wrapper(galapagos_interface *in_13, galapagos_interface *out_0);
#endif