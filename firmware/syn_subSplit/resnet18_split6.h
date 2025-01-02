#ifndef RESNET18_SPLIT6_H_
#define RESNET18_SPLIT6_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void kernel6(
    hls::stream<layer53_t> (&layer53_out)[N_FILT_40/256],
    hls::stream<layer56_t> (&layer56_out)[N_FILT_54/256], hls::stream<layer53_t> (&layer81_cpy2)[N_FILT_40/256]
);

#endif
