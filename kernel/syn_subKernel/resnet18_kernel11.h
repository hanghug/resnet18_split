#ifndef RESNET18_KERNEL11_H_
#define RESNET18_KERNEL11_H_

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

void kernel11_wrapper(galapagos_interface *in_13, galapagos_interface *out_0);
#endif