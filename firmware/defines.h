#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 224
#define N_INPUT_2_1 224
#define N_INPUT_3_1 3
#define OUT_HEIGHT_83 229
#define OUT_WIDTH_83 229
#define N_CHAN_83 3
#define OUT_HEIGHT_2 112
#define OUT_WIDTH_2 112
#define N_FILT_2 64
#define OUT_HEIGHT_5 56
#define OUT_WIDTH_5 56
#define N_FILT_5 64
#define OUT_HEIGHT_84 58
#define OUT_WIDTH_84 58
#define N_CHAN_84 64
#define OUT_HEIGHT_6 56
#define OUT_WIDTH_6 56
#define N_FILT_6 64
#define OUT_HEIGHT_85 58
#define OUT_WIDTH_85 58
#define N_CHAN_85 64
#define OUT_HEIGHT_9 56
#define OUT_WIDTH_9 56
#define N_FILT_9 64
#define OUT_HEIGHT_86 58
#define OUT_WIDTH_86 58
#define N_CHAN_86 64
#define OUT_HEIGHT_13 56
#define OUT_WIDTH_13 56
#define N_FILT_13 64
#define OUT_HEIGHT_87 58
#define OUT_WIDTH_87 58
#define N_CHAN_87 64
#define OUT_HEIGHT_16 56
#define OUT_WIDTH_16 56
#define N_FILT_16 64
#define OUT_HEIGHT_88 57
#define OUT_WIDTH_88 57
#define N_CHAN_88 64
#define OUT_HEIGHT_20 28
#define OUT_WIDTH_20 28
#define N_FILT_20 128
#define OUT_HEIGHT_89 30
#define OUT_WIDTH_89 30
#define N_CHAN_89 128
#define OUT_HEIGHT_100 28
#define OUT_WIDTH_100 28
#define N_FILT_100 128
#define OUT_HEIGHT_25 28
#define OUT_WIDTH_25 28
#define N_FILT_25 128
#define OUT_HEIGHT_23 28
#define OUT_WIDTH_23 28
#define N_FILT_23 128
#define OUT_HEIGHT_90 30
#define OUT_WIDTH_90 30
#define N_CHAN_90 128
#define OUT_HEIGHT_30 28
#define OUT_WIDTH_30 28
#define N_FILT_30 128
#define OUT_HEIGHT_91 30
#define OUT_WIDTH_91 30
#define N_CHAN_91 128
#define OUT_HEIGHT_33 28
#define OUT_WIDTH_33 28
#define N_FILT_33 128
#define OUT_HEIGHT_92 29
#define OUT_WIDTH_92 29
#define N_CHAN_92 128
#define OUT_HEIGHT_37 14
#define OUT_WIDTH_37 14
#define N_FILT_37 256
#define OUT_HEIGHT_93 16
#define OUT_WIDTH_93 16
#define N_CHAN_93 256
#define OUT_HEIGHT_101 14
#define OUT_WIDTH_101 14
#define N_FILT_101 256
#define OUT_HEIGHT_42 14
#define OUT_WIDTH_42 14
#define N_FILT_42 256
#define OUT_HEIGHT_40 14
#define OUT_WIDTH_40 14
#define N_FILT_40 256
#define OUT_HEIGHT_94 16
#define OUT_WIDTH_94 16
#define N_CHAN_94 256
#define OUT_HEIGHT_47 14
#define OUT_WIDTH_47 14
#define N_FILT_47 256
#define OUT_HEIGHT_95 16
#define OUT_WIDTH_95 16
#define N_CHAN_95 256
#define OUT_HEIGHT_50 14
#define OUT_WIDTH_50 14
#define N_FILT_50 256
#define OUT_HEIGHT_96 15
#define OUT_WIDTH_96 15
#define N_CHAN_96 256
#define OUT_HEIGHT_54 7
#define OUT_WIDTH_54 7
#define N_FILT_54 512
#define OUT_HEIGHT_97 9
#define OUT_WIDTH_97 9
#define N_CHAN_97 512
#define OUT_HEIGHT_102 7
#define OUT_WIDTH_102 7
#define N_FILT_102 512
#define OUT_HEIGHT_59 7
#define OUT_WIDTH_59 7
#define N_FILT_59 512
#define N_FILT_73 256     // ! q_conv2d_batchnorm_16输入的拆分，减半通道数。
#define OUT_HEIGHT_57 7
#define OUT_WIDTH_57 7
#define N_FILT_57 512
#define OUT_HEIGHT_98 9
#define OUT_WIDTH_98 9
#define N_CHAN_98 512
#define OUT_HEIGHT_64 7
#define OUT_WIDTH_64 7
#define N_FILT_64 512
#define N_FILT_74 256     // ! q_conv2d_batchnorm_18输入的拆分，减半通道数。
#define OUT_HEIGHT_99 9
#define OUT_WIDTH_99 9
#define N_CHAN_99 512
#define OUT_HEIGHT_67 7
#define OUT_WIDTH_67 7
#define N_FILT_67 512
#define N_FILT_75 256     // ! q_conv2d_batchnorm_19输入的拆分，减半通道数。
#define N_FILT_71 512
#define N_LAYER_72 1000


// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<8,2>, 3> input_t;
typedef nnet::array<ap_fixed<8,2>, 3> layer83_t;
typedef ap_fixed<8,2> q_conv2d_batchnorm_accum_t;
typedef nnet::array<ap_fixed<8,2>, 64> layer2_t;
typedef ap_fixed<8,2> weight2_t;
typedef ap_fixed<8,2> bias2_t;
typedef nnet::array<ap_fixed<8,2>, 64> layer4_t;
typedef ap_fixed<18,8> activation_table_t;
typedef ap_fixed<8,4> max_pooling2d_accum_t;
typedef nnet::array<ap_fixed<8,2>, 64> layer5_t;
typedef nnet::array<ap_fixed<8,2>, 64> layer84_t;
typedef ap_fixed<8,2> q_conv2d_batchnorm_1_accum_t;
typedef nnet::array<ap_fixed<8,2>, 64> layer6_t;
typedef ap_fixed<8,2> weight6_t;
typedef ap_fixed<8,2> bias6_t;
typedef nnet::array<ap_fixed<8,2>, 64> layer8_t;
typedef ap_fixed<18,8> activation_1_table_t;
typedef nnet::array<ap_fixed<8,2>, 64> layer85_t;
typedef ap_fixed<8,2> q_conv2d_batchnorm_2_accum_t;
typedef nnet::array<ap_fixed<8,2>, 64> layer9_t;
typedef ap_fixed<8,2> weight9_t;
typedef ap_fixed<8,2> bias9_t;
typedef nnet::array<ap_fixed<8,2>, 64> layer11_t;
typedef ap_fixed<18,8> activation_2_table_t;
typedef nnet::array<ap_fixed<8,2>, 64> layer12_t;
typedef nnet::array<ap_fixed<8,2>, 64> layer86_t;
typedef ap_fixed<8,2> q_conv2d_batchnorm_3_accum_t;
typedef nnet::array<ap_fixed<8,2>, 64> layer13_t;
typedef ap_fixed<8,2> weight13_t;
typedef ap_fixed<8,2> bias13_t;
typedef nnet::array<ap_fixed<8,2>, 64> layer15_t;
typedef ap_fixed<18,8> activation_3_table_t;
typedef nnet::array<ap_fixed<8,2>, 64> layer87_t;
typedef ap_fixed<8,2> q_conv2d_batchnorm_4_accum_t;
typedef nnet::array<ap_fixed<8,2>, 64> layer16_t;
typedef ap_fixed<8,2> weight16_t;
typedef ap_fixed<8,2> bias16_t;
typedef nnet::array<ap_fixed<8,2>, 64> layer18_t;
typedef ap_fixed<18,8> activation_4_table_t;
typedef nnet::array<ap_fixed<8,2>, 64> layer19_t;
typedef nnet::array<ap_fixed<8,2>, 64> layer88_t;
typedef ap_fixed<8,2> q_conv2d_batchnorm_5_accum_t;
typedef nnet::array<ap_fixed<8,2>, 128> layer20_t;
typedef ap_fixed<8,2> weight20_t;
typedef ap_fixed<8,2> bias20_t;
typedef nnet::array<ap_fixed<8,2>, 128> layer22_t;
typedef ap_fixed<18,8> activation_5_table_t;
typedef nnet::array<ap_fixed<8,2>, 128> layer89_t;
typedef ap_fixed<8,2> q_conv2d_batchnorm_7_accum_t;
typedef nnet::array<ap_fixed<8,2>, 128> layer100_t;
typedef ap_fixed<8,2> weight100_t;
typedef ap_fixed<8,2> bias100_t;
typedef ap_fixed<8,2> q_conv2d_batchnorm_6_accum_t;
typedef nnet::array<ap_fixed<8,2>, 128> layer25_t;
typedef ap_fixed<8,2> weight25_t;
typedef ap_fixed<8,2> bias25_t;
typedef nnet::array<ap_fixed<8,2>, 128> layer27_t;
typedef ap_fixed<18,8> activation_7_table_t;
typedef nnet::array<ap_fixed<8,2>, 128> layer28_t;
typedef ap_fixed<18,8> activation_6_table_t;
typedef nnet::array<ap_fixed<8,2>, 128> layer29_t;
typedef nnet::array<ap_fixed<8,2>, 128> layer90_t;
typedef ap_fixed<8,2> q_conv2d_batchnorm_8_accum_t;
typedef nnet::array<ap_fixed<8,2>, 128> layer30_t;
typedef ap_fixed<8,2> weight30_t;
typedef ap_fixed<8,2> bias30_t;
typedef nnet::array<ap_fixed<8,2>, 128> layer32_t;
typedef ap_fixed<18,8> activation_8_table_t;
typedef nnet::array<ap_fixed<8,2>, 128> layer91_t;
typedef ap_fixed<8,2> q_conv2d_batchnorm_9_accum_t;
typedef nnet::array<ap_fixed<8,2>, 128> layer33_t;
typedef ap_fixed<8,2> weight33_t;
typedef ap_fixed<8,2> bias33_t;
typedef nnet::array<ap_fixed<8,2>, 128> layer35_t;
typedef ap_fixed<18,8> activation_9_table_t;
typedef nnet::array<ap_fixed<8,2>, 128> layer36_t;
typedef nnet::array<ap_fixed<8,2>, 128> layer92_t;
typedef ap_fixed<8,2> q_conv2d_batchnorm_10_accum_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer37_t;
typedef ap_fixed<8,2> weight37_t;
typedef ap_fixed<8,2> bias37_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer39_t;
typedef ap_fixed<18,8> activation_10_table_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer93_t;
typedef ap_fixed<8,2> q_conv2d_batchnorm_12_accum_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer101_t;
typedef ap_fixed<8,2> weight101_t;
typedef ap_fixed<8,2> bias101_t;
typedef ap_fixed<8,2> q_conv2d_batchnorm_11_accum_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer42_t;
typedef ap_fixed<8,2> weight42_t;
typedef ap_fixed<8,2> bias42_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer44_t;
typedef ap_fixed<18,8> activation_12_table_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer45_t;
typedef ap_fixed<18,8> activation_11_table_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer46_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer94_t;
typedef ap_fixed<8,2> q_conv2d_batchnorm_13_accum_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer47_t;
typedef ap_fixed<8,2> weight47_t;
typedef ap_fixed<8,2> bias47_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer49_t;
typedef ap_fixed<18,8> activation_13_table_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer95_t;
typedef ap_fixed<8,2> q_conv2d_batchnorm_14_accum_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer50_t;
typedef ap_fixed<8,2> weight50_t;
typedef ap_fixed<8,2> bias50_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer52_t;
typedef ap_fixed<18,8> activation_14_table_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer53_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer96_t;
typedef ap_fixed<8,2> q_conv2d_batchnorm_15_accum_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer54_t;
typedef ap_fixed<8,2> weight54_t;
typedef ap_fixed<8,2> bias54_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer56_t;
typedef ap_fixed<18,8> activation_15_table_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer97_t;
typedef ap_fixed<8,2> q_conv2d_batchnorm_17_accum_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer102_t;
typedef ap_fixed<8,2> weight102_t;
typedef ap_fixed<8,2> bias102_t;
typedef ap_fixed<8,2> q_conv2d_batchnorm_16_accum_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer59_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer73_t;   //! q_conv2d_batchnorm_16输入的拆分,避免和layer59_t混淆，虽然在这里都一样。
typedef ap_fixed<8,2> weight59_t;
typedef ap_fixed<8,2> bias59_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer61_t;
typedef ap_fixed<18,8> activation_17_table_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer62_t;
typedef ap_fixed<18,8> activation_16_table_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer63_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer98_t;
typedef ap_fixed<8,2> q_conv2d_batchnorm_18_accum_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer74_t;   //! 
typedef nnet::array<ap_fixed<8,2>, 256> layer64_t;
typedef ap_fixed<8,2> weight64_t;
typedef ap_fixed<8,2> bias64_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer66_t;
typedef ap_fixed<18,8> activation_18_table_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer99_t;
typedef ap_fixed<8,2> q_conv2d_batchnorm_19_accum_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer67_t;
typedef ap_fixed<8,2> weight67_t;
typedef ap_fixed<8,2> bias67_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer69_t;
typedef ap_fixed<18,8> activation_19_table_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer70_t;
typedef ap_fixed<8,7> global_average_pooling2d_accum_t;
typedef nnet::array<ap_fixed<8,2>, 256> layer75_t;   //! 
typedef nnet::array<ap_fixed<8,2>, 256> layer71_t;
typedef ap_fixed<8,2> q_dense_accum_t;
typedef nnet::array<ap_fixed<8,2>, 200> layer72_t;
typedef ap_fixed<8,2> weight72_t;
typedef ap_fixed<8,2> bias72_t;
typedef ap_uint<1> layer72_index;
typedef nnet::array<ap_fixed<8,2>, 200> result_t;
typedef ap_fixed<18,8> activation_20_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT> activation_20_exp_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT> activation_20_inv_table_t;

#endif
