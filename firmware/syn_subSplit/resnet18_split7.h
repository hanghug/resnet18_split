#ifndef RESNET18_SPLIT7_H_
#define RESNET18_SPLIT7_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void kernel7(
    hls::stream<layer56_t> (&layer56_out)[N_FILT_54/256],
    hls::stream<layer62_t> (&layer62_out)[N_FILT_59/256]
);

#endif
