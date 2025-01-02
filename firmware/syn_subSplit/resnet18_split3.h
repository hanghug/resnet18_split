#ifndef RESNET18_SPLIT3_H_
#define RESNET18_SPLIT3_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void kernel3(
    hls::stream<layer29_t> (&layer29_out)[N_FILT_23/128],
    hls::stream<layer36_t> (&layer36_out)[N_FILT_23/128]
);

#endif
