#ifndef RESNET18_SPLIT1_H_
#define RESNET18_SPLIT1_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void kernel1(
    hls::stream<input_t> (&input_1)[N_INPUT_3_1/3],
    hls::stream<layer12_t> (&layer12_out)[N_FILT_5/64]
);

#endif
