#ifndef RESNET18_SPLIT10_H_
#define RESNET18_SPLIT10_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void kernel10(
    hls::stream<layer66_t> (&layer66_out)[N_FILT_64/256], hls::stream<layer63_t> (&layer82_cpy2)[N_FILT_57/256],
    hls::stream<layer70_t> (&layer70_out)[N_FILT_57/256]
);

#endif
