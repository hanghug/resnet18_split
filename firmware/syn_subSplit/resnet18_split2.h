#ifndef RESNET18_SPLIT2_H_
#define RESNET18_SPLIT2_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void kernel2(
    hls::stream<layer12_t> (&layer12_out)[N_FILT_5/64],
    hls::stream<layer29_t> (&layer29_out)[N_FILT_23/128]
);

#endif
