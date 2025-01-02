#ifndef RESNET18_SPLIT11_H_
#define RESNET18_SPLIT11_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void kernel11(
    hls::stream<layer70_t> (&layer70_out)[N_FILT_57/256],
    hls::stream<result_t> (&layer74_out)[N_LAYER_72/200]
);

#endif
