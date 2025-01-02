#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "firmware/resnet18_split.h"
#include "firmware/nnet_utils/nnet_helpers.h"

// hls-fpga-machine-learning insert bram

#define CHECKPOINT 1

namespace nnet {
bool trace_enabled = true;
std::map<std::string, void *> *trace_outputs = NULL;
size_t trace_type_size = sizeof(double);
} // namespace nnet

int main(int argc, char **argv) {
    // load input data from text file
    std::ifstream fin("tb_data/tb_input_features.dat");
    // load predictions from text file
    std::ifstream fpr("tb_data/tb_output_predictions.dat");

#ifdef RTL_SIM
    std::string RESULTS_LOG = "tb_data/rtl_cosim_results.log";
#else
    std::string RESULTS_LOG = "tb_data/csim_results.log";
#endif
    std::ofstream fout(RESULTS_LOG);

    std::string iline;
    std::string pline;
    int e = 0;

    if (fin.is_open() && fpr.is_open()) {
        while (std::getline(fin, iline) && std::getline(fpr, pline)) {
            if (e % CHECKPOINT == 0)
                std::cout << "Processing input " << e << std::endl;
            char *cstr = const_cast<char *>(iline.c_str());
            char *current;
            std::vector<float> in;
            current = strtok(cstr, ",");
            while (current != NULL) {
                in.push_back(atof(current));
                current = strtok(NULL, ",");
            }
            cstr = const_cast<char *>(pline.c_str());
            std::vector<float> pr;
            current = strtok(cstr, ",");
            while (current != NULL) {
                pr.push_back(atof(current));
                current = strtok(NULL, ",");
            }

            // hls-fpga-machine-learning insert data
      hls::stream<input_t> input_1[N_INPUT_3_1/3];
      nnet::copy_data<float, input_t, 0, N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1, N_INPUT_3_1>(in, input_1);
      hls::stream<result_t> layer74_out[N_LAYER_72/200];
      hls::stream<layer12_t> layer12_out[N_FILT_5/64];
      hls::stream<layer29_t> layer29_out[N_FILT_23/128];
      hls::stream<layer36_t> layer36_out[N_FILT_23/128];
      hls::stream<layer46_t> layer46_out[N_FILT_40/256];
      hls::stream<layer53_t> layer53_out[N_FILT_40/256];
      hls::stream<layer56_t> layer56_out[N_FILT_54/256];
      hls::stream<layer53_t> layer81_cpy2[N_FILT_40/256];
      hls::stream<layer62_t> layer62_out[N_FILT_59/256];
      hls::stream<layer61_t> layer61_out[N_FILT_57/256];
      hls::stream<layer66_t> layer66_out[N_FILT_64/256];
      hls::stream<layer63_t> layer82_cpy2[N_FILT_57/256];
      hls::stream<layer70_t> layer70_out[N_FILT_57/256];

            // hls-fpga-machine-learning insert top-level-function
            kernel1(input_1,layer12_out);
            kernel2(layer12_out,layer29_out);
            kernel3(layer29_out,layer36_out);
            kernel4(layer36_out,layer46_out);
            kernel5(layer46_out,layer53_out);
            kernel6(layer53_out,layer56_out,layer81_cpy2);
            kernel7(layer56_out,layer62_out);
            kernel8(layer81_cpy2,layer61_out);
            kernel9(layer61_out,layer62_out,layer66_out,layer82_cpy2);
            kernel10(layer66_out,layer82_cpy2,layer70_out);
            kernel11(layer70_out,layer74_out);

            if (e % CHECKPOINT == 0) {
                std::cout << "Predictions" << std::endl;
                // hls-fpga-machine-learning insert predictions
                for(int i = 0; i < N_LAYER_72; i++) {
                  std::cout << pr[i] << " ";
                }
                std::cout << std::endl;
                std::cout << "Quantized predictions" << std::endl;
                // hls-fpga-machine-learning insert quantized
                nnet::print_result<result_t, N_LAYER_72, N_LAYER_72>(layer74_out, std::cout, true);
            }
            e++;

            // hls-fpga-machine-learning insert tb-output
            nnet::print_result<result_t, N_LAYER_72, N_LAYER_72>(layer74_out, fout);
        }
        fin.close();
        fpr.close();
    } else {
        std::cout << "INFO: Unable to open input/predictions file, using default input." << std::endl;

        // hls-fpga-machine-learning insert zero
    hls::stream<input_t> input_1[N_INPUT_3_1/3];
    nnet::fill_zero<input_t, N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1, N_INPUT_3_1>(input_1);
    hls::stream<result_t> layer74_out[N_LAYER_72/200];
    hls::stream<layer12_t> layer12_out[N_FILT_5/64];
    hls::stream<layer29_t> layer29_out[N_FILT_23/128];
    hls::stream<layer36_t> layer36_out[N_FILT_23/128];
    hls::stream<layer46_t> layer46_out[N_FILT_40/256];
    hls::stream<layer53_t> layer53_out[N_FILT_40/256];
    hls::stream<layer56_t> layer56_out[N_FILT_54/256];
    hls::stream<layer53_t> layer81_cpy2[N_FILT_40/256];
    hls::stream<layer62_t> layer62_out[N_FILT_59/256];
    hls::stream<layer61_t> layer61_out[N_FILT_57/256];
    hls::stream<layer66_t> layer66_out[N_FILT_64/256];
    hls::stream<layer63_t> layer82_cpy2[N_FILT_57/256];
    hls::stream<layer70_t> layer70_out[N_FILT_57/256];

        // hls-fpga-machine-learning insert top-level-function
        kernel1(input_1,layer12_out);
        kernel2(layer12_out,layer29_out);
        kernel3(layer29_out,layer36_out);
        kernel4(layer36_out,layer46_out);
        kernel5(layer46_out,layer53_out);
        kernel6(layer53_out,layer56_out,layer81_cpy2);
        kernel7(layer56_out,layer62_out);
        kernel8(layer81_cpy2,layer61_out);
        kernel9(layer61_out,layer62_out,layer66_out,layer82_cpy2);
        kernel10(layer66_out,layer82_cpy2,layer70_out);
        kernel11(layer70_out,layer74_out);

        // hls-fpga-machine-learning insert output
        nnet::print_result<result_t, N_LAYER_72, N_LAYER_72>(layer74_out, std::cout, true);

        // hls-fpga-machine-learning insert tb-output
        nnet::print_result<result_t, N_LAYER_72, N_LAYER_72>(layer74_out, fout);
    }

    fout.close();
    std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

    return 0;
}
