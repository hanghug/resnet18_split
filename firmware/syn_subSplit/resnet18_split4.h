#ifndef RESNET18_SPLIT4_H_
#define RESNET18_SPLIT4_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void kernel4(
    hls::stream<layer36_t> (&layer36_out)[N_FILT_23/128],
    hls::stream<layer46_t> (&layer46_out)[N_FILT_40/256]
);

#endif
