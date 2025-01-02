#ifndef RESNET18_SPLIT8_H_
#define RESNET18_SPLIT8_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void kernel8(
    hls::stream<layer53_t> (&layer81_cpy2)[N_FILT_40/256],
    hls::stream<layer61_t> (&layer61_out)[N_FILT_57/256]
);

#endif
