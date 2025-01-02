#ifndef RESNET18_SPLIT_H_
#define RESNET18_SPLIT_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void kernel1(
    hls::stream<input_t> (&input_1)[N_INPUT_3_1/3],
    hls::stream<layer12_t> (&layer12_out)[N_FILT_5/64]
);


void kernel2(
    hls::stream<layer12_t> (&layer12_out)[N_FILT_5/64],
    hls::stream<layer29_t> (&layer29_out)[N_FILT_23/128]
);


void kernel3(
    hls::stream<layer29_t> (&layer29_out)[N_FILT_23/128],
    hls::stream<layer36_t> (&layer36_out)[N_FILT_23/128]
);


void kernel4(
    hls::stream<layer36_t> (&layer36_out)[N_FILT_23/128],
    hls::stream<layer46_t> (&layer46_out)[N_FILT_40/256]
);


void kernel5(
    hls::stream<layer46_t> (&layer46_out)[N_FILT_40/256],
    hls::stream<layer53_t> (&layer53_out)[N_FILT_40/256]
);


void kernel6(
    hls::stream<layer53_t> (&layer53_out)[N_FILT_40/256],
    hls::stream<layer56_t> (&layer56_out)[N_FILT_54/256], hls::stream<layer53_t> (&layer81_cpy2)[N_FILT_40/256]
);


void kernel7(
    hls::stream<layer56_t> (&layer56_out)[N_FILT_54/256],
    hls::stream<layer62_t> (&layer62_out)[N_FILT_59/256]
);


void kernel8(
    hls::stream<layer53_t> (&layer81_cpy2)[N_FILT_40/256],
    hls::stream<layer61_t> (&layer61_out)[N_FILT_57/256]
);


void kernel9(
    hls::stream<layer61_t> (&layer61_out)[N_FILT_57/256], hls::stream<layer62_t> (&layer62_out)[N_FILT_59/256],
    hls::stream<layer66_t> (&layer66_out)[N_FILT_64/256], hls::stream<layer63_t> (&layer82_cpy2)[N_FILT_57/256]
);


void kernel10(
    hls::stream<layer66_t> (&layer66_out)[N_FILT_64/256], hls::stream<layer63_t> (&layer82_cpy2)[N_FILT_57/256],
    hls::stream<layer70_t> (&layer70_out)[N_FILT_57/256]
);


void kernel11(
    hls::stream<layer70_t> (&layer70_out)[N_FILT_57/256],
    hls::stream<result_t> (&layer74_out)[N_LAYER_72/200]
);

#endif
