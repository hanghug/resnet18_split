#ifndef RESNET18_SPLIT5_H_
#define RESNET18_SPLIT5_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void kernel5(
    hls::stream<layer46_t> (&layer46_out)[N_FILT_40/256],
    hls::stream<layer53_t> (&layer53_out)[N_FILT_40/256]
);

#endif
