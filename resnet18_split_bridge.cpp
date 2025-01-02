#ifndef RESNET18_SPLIT_BRIDGE_H_
#define RESNET18_SPLIT_BRIDGE_H_

#include "firmware/resnet18_split.h"
#include "firmware/nnet_utils/nnet_helpers.h"
#include <algorithm>
#include <map>

// hls-fpga-machine-learning insert bram

namespace nnet {
bool trace_enabled = false;
std::map<std::string, void *> *trace_outputs = NULL;
size_t trace_type_size = sizeof(double);
} // namespace nnet

extern "C" {

struct trace_data {
    const char *name;
    void *data;
};

void allocate_trace_storage(size_t element_size) {
    nnet::trace_enabled = true;
    nnet::trace_outputs = new std::map<std::string, void *>;
    nnet::trace_type_size = element_size;
}

void free_trace_storage() {
    for (std::map<std::string, void *>::iterator i = nnet::trace_outputs->begin(); i != nnet::trace_outputs->end(); i++) {
        void *ptr = i->second;
        free(ptr);
    }
    nnet::trace_outputs->clear();
    delete nnet::trace_outputs;
    nnet::trace_outputs = NULL;
    nnet::trace_enabled = false;
}

void collect_trace_output(struct trace_data *c_trace_outputs) {
    int ii = 0;
    for (std::map<std::string, void *>::iterator i = nnet::trace_outputs->begin(); i != nnet::trace_outputs->end(); i++) {
        c_trace_outputs[ii].name = i->first.c_str();
        c_trace_outputs[ii].data = i->second;
        ii++;
    }
}

// Wrapper of top level function for Python bridge
void resnet18_split_float(
    float input_1[N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1],
    float layer74_out[N_LAYER_72]
) {

    hls::stream<input_t> input_1_ap[N_INPUT_3_1/3];
    nnet::convert_data<float, input_t, N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1, N_INPUT_3_1>(input_1, input_1_ap);

    hls::stream<result_t> layer74_out_ap[N_LAYER_72/200];
    hls::stream<layer12_t> layer12_out_ap[N_FILT_5/64];
    hls::stream<layer29_t> layer29_out_ap[N_FILT_23/128];
    hls::stream<layer36_t> layer36_out_ap[N_FILT_23/128];
    hls::stream<layer46_t> layer46_out_ap[N_FILT_40/256];
    hls::stream<layer53_t> layer53_out_ap[N_FILT_40/256];
    hls::stream<layer56_t> layer56_out_ap[N_FILT_54/256];
    hls::stream<layer53_t> layer81_cpy2_ap[N_FILT_40/256];
    hls::stream<layer62_t> layer62_out_ap[N_FILT_59/256];
    hls::stream<layer61_t> layer61_out_ap[N_FILT_57/256];
    hls::stream<layer66_t> layer66_out_ap[N_FILT_64/256];
    hls::stream<layer63_t> layer82_cpy2_ap[N_FILT_57/256];
    hls::stream<layer70_t> layer70_out_ap[N_FILT_57/256];

    kernel1(input_1_ap,layer12_out_ap);
    kernel2(layer12_out_ap,layer29_out_ap);
    kernel3(layer29_out_ap,layer36_out_ap);
    kernel4(layer36_out_ap,layer46_out_ap);
    kernel5(layer46_out_ap,layer53_out_ap);
    kernel6(layer53_out_ap,layer56_out_ap,layer81_cpy2_ap);
    kernel7(layer56_out_ap,layer62_out_ap);
    kernel8(layer81_cpy2_ap,layer61_out_ap);
    kernel9(layer61_out_ap,layer62_out_ap,layer66_out_ap,layer82_cpy2_ap);
    kernel10(layer66_out_ap,layer82_cpy2_ap,layer70_out_ap);
    kernel11(layer70_out_ap,layer74_out_ap);

    nnet::convert_data<result_t, float, N_LAYER_72, N_LAYER_72>(layer74_out_ap, layer74_out);
}

void resnet18_split_double(
    double input_1[N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1],
    double layer74_out[N_LAYER_72]
) {
    hls::stream<input_t> input_1_ap[N_INPUT_3_1/3];
    nnet::convert_data<double, input_t, N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1, N_INPUT_3_1>(input_1, input_1_ap);

    hls::stream<result_t> layer74_out_ap[N_LAYER_72/200];
    hls::stream<layer12_t> layer12_out_ap[N_FILT_5/64];
    hls::stream<layer29_t> layer29_out_ap[N_FILT_23/128];
    hls::stream<layer36_t> layer36_out_ap[N_FILT_23/128];
    hls::stream<layer46_t> layer46_out_ap[N_FILT_40/256];
    hls::stream<layer53_t> layer53_out_ap[N_FILT_40/256];
    hls::stream<layer56_t> layer56_out_ap[N_FILT_54/256];
    hls::stream<layer53_t> layer81_cpy2_ap[N_FILT_40/256];
    hls::stream<layer62_t> layer62_out_ap[N_FILT_59/256];
    hls::stream<layer61_t> layer61_out_ap[N_FILT_57/256];
    hls::stream<layer66_t> layer66_out_ap[N_FILT_64/256];
    hls::stream<layer63_t> layer82_cpy2_ap[N_FILT_57/256];
    hls::stream<layer70_t> layer70_out_ap[N_FILT_57/256];

    kernel1(input_1_ap,layer12_out_ap);
    kernel2(layer12_out_ap,layer29_out_ap);
    kernel3(layer29_out_ap,layer36_out_ap);
    kernel4(layer36_out_ap,layer46_out_ap);
    kernel5(layer46_out_ap,layer53_out_ap);
    kernel6(layer53_out_ap,layer56_out_ap,layer81_cpy2_ap);
    kernel7(layer56_out_ap,layer62_out_ap);
    kernel8(layer81_cpy2_ap,layer61_out_ap);
    kernel9(layer61_out_ap,layer62_out_ap,layer66_out_ap,layer82_cpy2_ap);
    kernel10(layer66_out_ap,layer82_cpy2_ap,layer70_out_ap);
    kernel11(layer70_out_ap,layer74_out_ap);

    nnet::convert_data<result_t, double, N_LAYER_72, N_LAYER_72>(layer74_out_ap, layer74_out);
}
}

#endif
