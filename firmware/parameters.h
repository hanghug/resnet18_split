#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_conv2d.h"
#include "nnet_utils/nnet_conv2d_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_merge.h"
#include "nnet_utils/nnet_merge_stream.h"
#include "nnet_utils/nnet_padding.h"
#include "nnet_utils/nnet_padding_stream.h"
#include "nnet_utils/nnet_pooling.h"
#include "nnet_utils/nnet_pooling_stream.h"
#include "nnet_utils/nnet_sepconv2d_stream.h"
#include "nnet_utils/nnet_stream.h"

// hls-fpga-machine-learning insert weights

// hls-fpga-machine-learning insert layer-config
// zp2d_q_conv2d_batchnorm
struct config83 : nnet::padding2d_config {
    static const unsigned in_height = 224;
    static const unsigned in_width = 224;
    static const unsigned n_chan = 3;
    static const unsigned out_height = 229;
    static const unsigned out_width = 229;
    static const unsigned pad_top = 2;
    static const unsigned pad_bottom = 3;
    static const unsigned pad_left = 2;
    static const unsigned pad_right = 3;
};

// q_conv2d_batchnorm
struct config2_mult : nnet::dense_config {
    static const unsigned n_in = 147;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 1794;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_accum_t accum_t;
    typedef bias2_t bias_t;
    typedef weight2_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config2 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 229;
    static const unsigned in_width = 229;
    static const unsigned n_chan = 3;
    static const unsigned filt_height = 7;
    static const unsigned filt_width = 7;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 64;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned out_height = 112;
    static const unsigned out_width = 112;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 1794;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 13;
    static const unsigned min_width = 13;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 12544;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_accum_t accum_t;
    typedef bias2_t bias_t;
    typedef weight2_t weight_t;
    typedef config2_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config2::filt_height * config2::filt_width> config2::pixels[] = {1,2,5,10,21,42,85,42,84,40,80,32,64,128,256,640,1280,2688,5376,10880,5376,10752,5120,10240,4096,8192,16385,32770,81925,163850,344085,688170,1392725,688170,1376340,655400,1310800,524320,1048640,2097280,4194560,10486400,20972800,44042880,88085760,178268800,88085760,176171520,83891200,167782400,67112960,134225920,268451841,536903682,1342259205,2684518410,5637488661,11274977322,22818406485,11274977322,22549954644,10738073640,21476147280,8590458912,17180917824,34361835648,68723671296,171809178240,343618356480,721598548608,1443197097216,2920756030080,1443197097216,2886394194432,1374473425920,2748946851840,1099578740736,2199157481472,4398314962945,8796629925890,21991574814725,43983149629450,92364614221845,184729228443690,373856771850325,184729228443690,369458456887380,175932598517800,351865197035600,140746078814240,281492157628480,34361835648,68723671296,171809178240,343618356480,721598548608,1443197097216,2920756030080,1443197097216,2886394194432,1374473425920,2748946851840,1099578740736,2199157481472,4398314962944,8796629925888,21991574814720,43983149629440,92364614221824,184729228443648,373856771850240,184729228443648,369458456887296,175932598517760,351865197035520,140746078814208,281492157628416,34361835520,68723671040,171809177600,343618355200,721598545920,1443197091840,2920756019200,1443197091840,2886394183680,1374473420800,2748946841600,1099578736640,2199157473280,4398314946560,8796629893120,21991574732800,43983149465600,92364613877760,184729227755520,373856770457600,184729227755520,369458455511040,175932597862400,351865195724800,140746078289920,281492156579840,34359738368,68719476736,171798691840,343597383680,721554505728,1443109011456,2920577761280,1443109011456,2886218022912,1374389534720,2748779069440,1099511627776,2199023255552,4398046511104,8796093022208,21990232555520,43980465111040,92358976733184,184717953466368,373833953443840,184717953466368,369435906932736,175921860444160,351843720888320,140737488355328,281474976710656};

// activation
struct relu_config4 : nnet::activ_config {
    static const unsigned n_in = 802816;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef activation_table_t table_t;
};

// max_pooling2d
struct config5 : nnet::pooling2d_config {
    static const unsigned in_height = 112;
    static const unsigned in_width = 112;
    static const unsigned n_filt = 64;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = pool_height;
    static const unsigned filt_width = pool_width;
    static const unsigned n_chan = n_filt;

    static const unsigned out_height = 56;
    static const unsigned out_width = 56;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const bool count_pad = false;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse_factor = 1;
    typedef max_pooling2d_accum_t accum_t;
};

// zp2d_q_conv2d_batchnorm_1
struct config84 : nnet::padding2d_config {
    static const unsigned in_height = 56;
    static const unsigned in_width = 56;
    static const unsigned n_chan = 64;
    static const unsigned out_height = 58;
    static const unsigned out_width = 58;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_1
struct config6_mult : nnet::dense_config {
    static const unsigned n_in = 576;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 48;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 4134;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_1_accum_t accum_t;
    typedef bias6_t bias_t;
    typedef weight6_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config6 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 58;
    static const unsigned in_width = 58;
    static const unsigned n_chan = 64;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 64;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 56;
    static const unsigned out_width = 56;
    static const unsigned reuse_factor = 48;
    static const unsigned n_zeros = 4134;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 3136;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_1_accum_t accum_t;
    typedef bias6_t bias_t;
    typedef weight6_t weight_t;
    typedef config6_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config6::filt_height * config6::filt_width> config6::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// activation_1
struct relu_config8 : nnet::activ_config {
    static const unsigned n_in = 200704;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef activation_1_table_t table_t;
};

// zp2d_q_conv2d_batchnorm_2
struct config85 : nnet::padding2d_config {
    static const unsigned in_height = 56;
    static const unsigned in_width = 56;
    static const unsigned n_chan = 64;
    static const unsigned out_height = 58;
    static const unsigned out_width = 58;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_2
struct config9_mult : nnet::dense_config {
    static const unsigned n_in = 576;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 48;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 4134;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_2_accum_t accum_t;
    typedef bias9_t bias_t;
    typedef weight9_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config9 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 58;
    static const unsigned in_width = 58;
    static const unsigned n_chan = 64;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 64;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 56;
    static const unsigned out_width = 56;
    static const unsigned reuse_factor = 48;
    static const unsigned n_zeros = 4134;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 3136;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_2_accum_t accum_t;
    typedef bias9_t bias_t;
    typedef weight9_t weight_t;
    typedef config9_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config9::filt_height * config9::filt_width> config9::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// activation_2
struct relu_config11 : nnet::activ_config {
    static const unsigned n_in = 200704;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef activation_2_table_t table_t;
};

// add
struct config12 : nnet::merge_config {
    static const unsigned n_elem = OUT_HEIGHT_5*OUT_WIDTH_5*N_FILT_5;
};

// zp2d_q_conv2d_batchnorm_3
struct config86 : nnet::padding2d_config {
    static const unsigned in_height = 56;
    static const unsigned in_width = 56;
    static const unsigned n_chan = 64;
    static const unsigned out_height = 58;
    static const unsigned out_width = 58;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_3
struct config13_mult : nnet::dense_config {
    static const unsigned n_in = 576;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 48;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 4134;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_3_accum_t accum_t;
    typedef bias13_t bias_t;
    typedef weight13_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config13 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 58;
    static const unsigned in_width = 58;
    static const unsigned n_chan = 64;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 64;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 56;
    static const unsigned out_width = 56;
    static const unsigned reuse_factor = 48;
    static const unsigned n_zeros = 4134;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 3136;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_3_accum_t accum_t;
    typedef bias13_t bias_t;
    typedef weight13_t weight_t;
    typedef config13_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config13::filt_height * config13::filt_width> config13::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// activation_3
struct relu_config15 : nnet::activ_config {
    static const unsigned n_in = 200704;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef activation_3_table_t table_t;
};

// zp2d_q_conv2d_batchnorm_4
struct config87 : nnet::padding2d_config {
    static const unsigned in_height = 56;
    static const unsigned in_width = 56;
    static const unsigned n_chan = 64;
    static const unsigned out_height = 58;
    static const unsigned out_width = 58;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_4
struct config16_mult : nnet::dense_config {
    static const unsigned n_in = 576;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 48;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 4134;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_4_accum_t accum_t;
    typedef bias16_t bias_t;
    typedef weight16_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config16 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 58;
    static const unsigned in_width = 58;
    static const unsigned n_chan = 64;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 64;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 56;
    static const unsigned out_width = 56;
    static const unsigned reuse_factor = 48;
    static const unsigned n_zeros = 4134;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 3136;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_4_accum_t accum_t;
    typedef bias16_t bias_t;
    typedef weight16_t weight_t;
    typedef config16_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config16::filt_height * config16::filt_width> config16::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// activation_4
struct relu_config18 : nnet::activ_config {
    static const unsigned n_in = 200704;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef activation_4_table_t table_t;
};

// add_1
struct config19 : nnet::merge_config {
    static const unsigned n_elem = OUT_HEIGHT_5*OUT_WIDTH_5*N_FILT_5;
};

// zp2d_q_conv2d_batchnorm_5
struct config88 : nnet::padding2d_config {
    static const unsigned in_height = 56;
    static const unsigned in_width = 56;
    static const unsigned n_chan = 64;
    static const unsigned out_height = 57;
    static const unsigned out_width = 57;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_5
struct config20_mult : nnet::dense_config {
    static const unsigned n_in = 576;
    static const unsigned n_out = 128;
    static const unsigned reuse_factor = 48;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 9897;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_5_accum_t accum_t;
    typedef bias20_t bias_t;
    typedef weight20_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config20 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 57;
    static const unsigned in_width = 57;
    static const unsigned n_chan = 64;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 128;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned out_height = 28;
    static const unsigned out_width = 28;
    static const unsigned reuse_factor = 48;
    static const unsigned n_zeros = 9897;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 784;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_5_accum_t accum_t;
    typedef bias20_t bias_t;
    typedef weight20_t weight_t;
    typedef config20_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config20::filt_height * config20::filt_width> config20::pixels[] = {1,2,5,2,4,8,16,40,16,32,65,130,325,130,260,8,16,40,16,32,64,128,320,128,256};

// activation_5
struct relu_config22 : nnet::activ_config {
    static const unsigned n_in = 100352;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef activation_5_table_t table_t;
};

// zp2d_q_conv2d_batchnorm_6
struct config89 : nnet::padding2d_config {
    static const unsigned in_height = 28;
    static const unsigned in_width = 28;
    static const unsigned n_chan = 128;
    static const unsigned out_height = 30;
    static const unsigned out_width = 30;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_7
struct config100_mult : nnet::dense_config {
    static const unsigned n_in = 64;
    static const unsigned n_out = 128;
    static const unsigned reuse_factor = 32;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 396;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_7_accum_t accum_t;
    typedef bias100_t bias_t;
    typedef weight100_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config100 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 56;
    static const unsigned in_width = 56;
    static const unsigned n_chan = 64;
    static const unsigned filt_height = 1;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 128;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned out_height = 28;
    static const unsigned out_width = 28;
    static const unsigned reuse_factor = 32;
    static const unsigned n_zeros = 396;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 4;
    static const unsigned min_width = 4;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 784;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_7_accum_t accum_t;
    typedef bias100_t bias_t;
    typedef weight100_t weight_t;
    typedef config100_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config100::filt_height * config100::filt_width> config100::pixels[] = {1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0};

// q_conv2d_batchnorm_6
struct config25_mult : nnet::dense_config {
    static const unsigned n_in = 1152;
    static const unsigned n_out = 128;
    static const unsigned reuse_factor = 192;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 22596;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_6_accum_t accum_t;
    typedef bias25_t bias_t;
    typedef weight25_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config25 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 30;
    static const unsigned in_width = 30;
    static const unsigned n_chan = 128;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 128;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 28;
    static const unsigned out_width = 28;
    static const unsigned reuse_factor = 192;
    static const unsigned n_zeros = 22596;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 784;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_6_accum_t accum_t;
    typedef bias25_t bias_t;
    typedef weight25_t weight_t;
    typedef config25_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config25::filt_height * config25::filt_width> config25::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// activation_7
struct relu_config27 : nnet::activ_config {
    static const unsigned n_in = 100352;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef activation_7_table_t table_t;
};

// activation_6
struct relu_config28 : nnet::activ_config {
    static const unsigned n_in = 100352;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef activation_6_table_t table_t;
};

// add_2
struct config29 : nnet::merge_config {
    static const unsigned n_elem = OUT_HEIGHT_23*OUT_WIDTH_23*N_FILT_23;
};

// zp2d_q_conv2d_batchnorm_8
struct config90 : nnet::padding2d_config {
    static const unsigned in_height = 28;
    static const unsigned in_width = 28;
    static const unsigned n_chan = 128;
    static const unsigned out_height = 30;
    static const unsigned out_width = 30;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_8
struct config30_mult : nnet::dense_config {
    static const unsigned n_in = 1152;
    static const unsigned n_out = 128;
    static const unsigned reuse_factor = 192;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 22596;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_8_accum_t accum_t;
    typedef bias30_t bias_t;
    typedef weight30_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config30 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 30;
    static const unsigned in_width = 30;
    static const unsigned n_chan = 128;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 128;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 28;
    static const unsigned out_width = 28;
    static const unsigned reuse_factor = 192;
    static const unsigned n_zeros = 22596;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 784;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_8_accum_t accum_t;
    typedef bias30_t bias_t;
    typedef weight30_t weight_t;
    typedef config30_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config30::filt_height * config30::filt_width> config30::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// activation_8
struct relu_config32 : nnet::activ_config {
    static const unsigned n_in = 100352;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef activation_8_table_t table_t;
};

// zp2d_q_conv2d_batchnorm_9
struct config91 : nnet::padding2d_config {
    static const unsigned in_height = 28;
    static const unsigned in_width = 28;
    static const unsigned n_chan = 128;
    static const unsigned out_height = 30;
    static const unsigned out_width = 30;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_9
struct config33_mult : nnet::dense_config {
    static const unsigned n_in = 1152;
    static const unsigned n_out = 128;
    static const unsigned reuse_factor = 192;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 22596;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_9_accum_t accum_t;
    typedef bias33_t bias_t;
    typedef weight33_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config33 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 30;
    static const unsigned in_width = 30;
    static const unsigned n_chan = 128;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 128;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 28;
    static const unsigned out_width = 28;
    static const unsigned reuse_factor = 192;
    static const unsigned n_zeros = 22596;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 784;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_9_accum_t accum_t;
    typedef bias33_t bias_t;
    typedef weight33_t weight_t;
    typedef config33_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config33::filt_height * config33::filt_width> config33::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// activation_9
struct relu_config35 : nnet::activ_config {
    static const unsigned n_in = 100352;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef activation_9_table_t table_t;
};

// add_3
struct config36 : nnet::merge_config {
    static const unsigned n_elem = OUT_HEIGHT_23*OUT_WIDTH_23*N_FILT_23;
};

// zp2d_q_conv2d_batchnorm_10
struct config92 : nnet::padding2d_config {
    static const unsigned in_height = 28;
    static const unsigned in_width = 28;
    static const unsigned n_chan = 128;
    static const unsigned out_height = 29;
    static const unsigned out_width = 29;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_10
struct config37_mult : nnet::dense_config {
    static const unsigned n_in = 1152;
    static const unsigned n_out = 256;
    static const unsigned reuse_factor = 192;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 55108;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_10_accum_t accum_t;
    typedef bias37_t bias_t;
    typedef weight37_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config37 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 29;
    static const unsigned in_width = 29;
    static const unsigned n_chan = 128;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 256;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned out_height = 14;
    static const unsigned out_width = 14;
    static const unsigned reuse_factor = 192;
    static const unsigned n_zeros = 55108;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 196;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_10_accum_t accum_t;
    typedef bias37_t bias_t;
    typedef weight37_t weight_t;
    typedef config37_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config37::filt_height * config37::filt_width> config37::pixels[] = {1,2,5,2,4,8,16,40,16,32,65,130,325,130,260,8,16,40,16,32,64,128,320,128,256};

// activation_10
struct relu_config39 : nnet::activ_config {
    static const unsigned n_in = 50176;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef activation_10_table_t table_t;
};

// zp2d_q_conv2d_batchnorm_11
struct config93 : nnet::padding2d_config {
    static const unsigned in_height = 14;
    static const unsigned in_width = 14;
    static const unsigned n_chan = 256;
    static const unsigned out_height = 16;
    static const unsigned out_width = 16;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_12
struct config101_mult : nnet::dense_config {
    static const unsigned n_in = 128;
    static const unsigned n_out = 256;
    static const unsigned reuse_factor = 128;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 2153;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_12_accum_t accum_t;
    typedef bias101_t bias_t;
    typedef weight101_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config101 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 28;
    static const unsigned in_width = 28;
    static const unsigned n_chan = 128;
    static const unsigned filt_height = 1;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 256;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned out_height = 14;
    static const unsigned out_width = 14;
    static const unsigned reuse_factor = 128;
    static const unsigned n_zeros = 2153;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 4;
    static const unsigned min_width = 4;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 196;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_12_accum_t accum_t;
    typedef bias101_t bias_t;
    typedef weight101_t weight_t;
    typedef config101_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config101::filt_height * config101::filt_width> config101::pixels[] = {1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0};

// q_conv2d_batchnorm_11
struct config42_mult : nnet::dense_config {
    static const unsigned n_in = 2304;
    static const unsigned n_out = 256;
    static const unsigned reuse_factor = 768;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 127932;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_11_accum_t accum_t;
    typedef bias42_t bias_t;
    typedef weight42_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config42 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 16;
    static const unsigned in_width = 16;
    static const unsigned n_chan = 256;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 256;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 14;
    static const unsigned out_width = 14;
    static const unsigned reuse_factor = 768;
    static const unsigned n_zeros = 127932;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 196;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_11_accum_t accum_t;
    typedef bias42_t bias_t;
    typedef weight42_t weight_t;
    typedef config42_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config42::filt_height * config42::filt_width> config42::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// activation_12
struct relu_config44 : nnet::activ_config {
    static const unsigned n_in = 50176;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef activation_12_table_t table_t;
};

// activation_11
struct relu_config45 : nnet::activ_config {
    static const unsigned n_in = 50176;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef activation_11_table_t table_t;
};

// add_4
struct config46 : nnet::merge_config {
    static const unsigned n_elem = OUT_HEIGHT_40*OUT_WIDTH_40*N_FILT_40;
};

// zp2d_q_conv2d_batchnorm_13
struct config94 : nnet::padding2d_config {
    static const unsigned in_height = 14;
    static const unsigned in_width = 14;
    static const unsigned n_chan = 256;
    static const unsigned out_height = 16;
    static const unsigned out_width = 16;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_13
struct config47_mult : nnet::dense_config {
    static const unsigned n_in = 2304;
    static const unsigned n_out = 256;
    static const unsigned reuse_factor = 768;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 127932;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_13_accum_t accum_t;
    typedef bias47_t bias_t;
    typedef weight47_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config47 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 16;
    static const unsigned in_width = 16;
    static const unsigned n_chan = 256;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 256;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 14;
    static const unsigned out_width = 14;
    static const unsigned reuse_factor = 768;
    static const unsigned n_zeros = 127932;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 196;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_13_accum_t accum_t;
    typedef bias47_t bias_t;
    typedef weight47_t weight_t;
    typedef config47_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config47::filt_height * config47::filt_width> config47::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// activation_13
struct relu_config49 : nnet::activ_config {
    static const unsigned n_in = 50176;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef activation_13_table_t table_t;
};

// zp2d_q_conv2d_batchnorm_14
struct config95 : nnet::padding2d_config {
    static const unsigned in_height = 14;
    static const unsigned in_width = 14;
    static const unsigned n_chan = 256;
    static const unsigned out_height = 16;
    static const unsigned out_width = 16;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_14
struct config50_mult : nnet::dense_config {
    static const unsigned n_in = 2304;
    static const unsigned n_out = 256;
    static const unsigned reuse_factor = 768;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 127932;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_14_accum_t accum_t;
    typedef bias50_t bias_t;
    typedef weight50_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config50 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 16;
    static const unsigned in_width = 16;
    static const unsigned n_chan = 256;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 256;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 14;
    static const unsigned out_width = 14;
    static const unsigned reuse_factor = 768;
    static const unsigned n_zeros = 127932;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 196;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_14_accum_t accum_t;
    typedef bias50_t bias_t;
    typedef weight50_t weight_t;
    typedef config50_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config50::filt_height * config50::filt_width> config50::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// activation_14
struct relu_config52 : nnet::activ_config {
    static const unsigned n_in = 50176;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef activation_14_table_t table_t;
};

// add_5
struct config53 : nnet::merge_config {
    static const unsigned n_elem = OUT_HEIGHT_40*OUT_WIDTH_40*N_FILT_40;
};

// zp2d_q_conv2d_batchnorm_15
struct config96 : nnet::padding2d_config {
    static const unsigned in_height = 14;
    static const unsigned in_width = 14;
    static const unsigned n_chan = 256;
    static const unsigned out_height = 15;
    static const unsigned out_width = 15;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_15
struct config54_mult : nnet::dense_config {
    static const unsigned n_in = 2304;
    static const unsigned n_out = 512;
    static const unsigned reuse_factor = 768;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 312753;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_15_accum_t accum_t;
    typedef bias54_t bias_t;
    typedef weight54_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config54 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 15;
    static const unsigned in_width = 15;
    static const unsigned n_chan = 256;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 512;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned out_height = 7;
    static const unsigned out_width = 7;
    static const unsigned reuse_factor = 768;
    static const unsigned n_zeros = 312753;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 49;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_15_accum_t accum_t;
    typedef bias54_t bias_t;
    typedef weight54_t weight_t;
    typedef config54_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config54::filt_height * config54::filt_width> config54::pixels[] = {1,2,5,2,4,8,16,40,16,32,65,130,325,130,260,8,16,40,16,32,64,128,320,128,256};

// activation_15
struct relu_config56 : nnet::activ_config {
    static const unsigned n_in = 25088;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef activation_15_table_t table_t;
};

// zp2d_q_conv2d_batchnorm_16
struct config97 : nnet::padding2d_config {
    static const unsigned in_height = 7;
    static const unsigned in_width = 7;
    static const unsigned n_chan = 512;
    static const unsigned out_height = 9;
    static const unsigned out_width = 9;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_17
struct config102_mult : nnet::dense_config {
    static const unsigned n_in = 256;
    static const unsigned n_out = 512;
    static const unsigned reuse_factor = 512;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 11744;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_17_accum_t accum_t;
    typedef bias102_t bias_t;
    typedef weight102_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config102 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 14;
    static const unsigned in_width = 14;
    static const unsigned n_chan = 256;
    static const unsigned filt_height = 1;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 512;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned out_height = 7;
    static const unsigned out_width = 7;
    static const unsigned reuse_factor = 512;
    static const unsigned n_zeros = 11744;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 4;
    static const unsigned min_width = 4;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 49;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_17_accum_t accum_t;
    typedef bias102_t bias_t;
    typedef weight102_t weight_t;
    typedef config102_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config102::filt_height * config102::filt_width> config102::pixels[] = {1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0};










// q_conv2d_batchnorm_16
struct config59_mult : nnet::dense_config {
    static const unsigned n_in = 4608/2;    // 新增
    static const unsigned n_out = 512;
    static const unsigned reuse_factor = 2304;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 721454;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_16_accum_t accum_t;
    typedef bias59_t bias_t;
    typedef weight59_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config59 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 9;
    static const unsigned in_width = 9;
    static const unsigned n_chan = 512/2;    // 新增
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 512;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 7;
    static const unsigned out_width = 7;
    static const unsigned reuse_factor = 2304;
    static const unsigned n_zeros = 721454;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 49;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_16_accum_t accum_t;
    typedef bias59_t bias_t;
    typedef weight59_t weight_t;
    typedef config59_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config59::filt_height * config59::filt_width> config59::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};



struct config59_mult_1 : nnet::dense_config {
    static const unsigned n_in = 4608/2;    // 新增
    static const unsigned n_out = 512;
    static const unsigned reuse_factor = 2304;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 721454;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_16_accum_t accum_t;
    typedef bias59_t bias_t;
    typedef weight59_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config59_1 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 9;
    static const unsigned in_width = 9;
    static const unsigned n_chan = 512/2;    // 新增
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 512;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 7;
    static const unsigned out_width = 7;
    static const unsigned reuse_factor = 2304;
    static const unsigned n_zeros = 721454;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 49;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_16_accum_t accum_t;
    typedef bias59_t bias_t;
    typedef weight59_t weight_t;
    typedef config59_mult_1 mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};

struct config59_mult_2 : nnet::dense_config {   //!
    static const unsigned n_in = 4608/2;    // 新增
    static const unsigned n_out = 512;
    static const unsigned reuse_factor = 2304;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 721454;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_16_accum_t accum_t;
    typedef bias59_t bias_t;
    typedef weight59_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config59_2 : nnet::conv2d_config {  //!
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 9;
    static const unsigned in_width = 9;
    static const unsigned n_chan = 512/2;    // 新增
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 512;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 7;
    static const unsigned out_width = 7;
    static const unsigned reuse_factor = 2304;
    static const unsigned n_zeros = 721454;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 49;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_16_accum_t accum_t;
    typedef bias59_t bias_t;
    typedef weight59_t weight_t;
    typedef config59_mult_2 mult_config;    //!
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};


struct config59_split {  //!
    static const unsigned in_height = 9;
    static const unsigned in_width = 9;
    static const unsigned n_chan = 512/2;    // 新增
};


struct config59_add {  //!
    static const unsigned n_filt = 512;
    static const unsigned out_height = 7;
    static const unsigned out_width = 7;
};






// activation_17
struct relu_config61 : nnet::activ_config {
    static const unsigned n_in = 25088;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef activation_17_table_t table_t;
};

// activation_16
struct relu_config62 : nnet::activ_config {
    static const unsigned n_in = 25088;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef activation_16_table_t table_t;
};

// add_6
struct config63 : nnet::merge_config {
    static const unsigned n_elem = OUT_HEIGHT_57*OUT_WIDTH_57*N_FILT_57;
};

// zp2d_q_conv2d_batchnorm_18
struct config98 : nnet::padding2d_config {
    static const unsigned in_height = 7;
    static const unsigned in_width = 7;
    static const unsigned n_chan = 512;
    static const unsigned out_height = 9;
    static const unsigned out_width = 9;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};









// q_conv2d_batchnorm_18
struct config64_mult : nnet::dense_config {
    static const unsigned n_in = 4608/2;   //!
    static const unsigned n_out = 512;
    static const unsigned reuse_factor = 2304;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 721454;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_18_accum_t accum_t;
    typedef bias64_t bias_t;
    typedef weight64_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config64 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 9;
    static const unsigned in_width = 9;
    static const unsigned n_chan = 512/2;   //! 
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 512;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 7;
    static const unsigned out_width = 7;
    static const unsigned reuse_factor = 2304;
    static const unsigned n_zeros = 721454;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 49;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_18_accum_t accum_t;
    typedef bias64_t bias_t;
    typedef weight64_t weight_t;
    typedef config64_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config64::filt_height * config64::filt_width> config64::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};



struct config64_mult_1 : nnet::dense_config {
    static const unsigned n_in = 4608/2;   //!
    static const unsigned n_out = 512;
    static const unsigned reuse_factor = 2304;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 721454;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_18_accum_t accum_t;
    typedef bias64_t bias_t;
    typedef weight64_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config64_1 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 9;
    static const unsigned in_width = 9;
    static const unsigned n_chan = 512/2;   //! 
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 512;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 7;
    static const unsigned out_width = 7;
    static const unsigned reuse_factor = 2304;
    static const unsigned n_zeros = 721454;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 49;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_18_accum_t accum_t;
    typedef bias64_t bias_t;
    typedef weight64_t weight_t;
    typedef config64_mult_1 mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};


struct config64_mult_2 : nnet::dense_config {
    static const unsigned n_in = 4608/2;   //!
    static const unsigned n_out = 512;
    static const unsigned reuse_factor = 2304;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 721454;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_18_accum_t accum_t;
    typedef bias64_t bias_t;
    typedef weight64_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config64_2 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 9;
    static const unsigned in_width = 9;
    static const unsigned n_chan = 512/2;   //! 
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 512;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 7;
    static const unsigned out_width = 7;
    static const unsigned reuse_factor = 2304;
    static const unsigned n_zeros = 721454;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 49;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_18_accum_t accum_t;
    typedef bias64_t bias_t;
    typedef weight64_t weight_t;
    typedef config64_mult_2 mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};

struct config64_split {
    static const unsigned in_height = 9;
    static const unsigned in_width = 9;
    static const unsigned n_chan = 512/2;   //! 
};

struct config64_add {
    static const unsigned n_filt = 512;
    static const unsigned out_height = 7;
    static const unsigned out_width = 7;
};






// activation_18
struct relu_config66 : nnet::activ_config {
    static const unsigned n_in = 25088;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef activation_18_table_t table_t;
};

// zp2d_q_conv2d_batchnorm_19
struct config99 : nnet::padding2d_config {
    static const unsigned in_height = 7;
    static const unsigned in_width = 7;
    static const unsigned n_chan = 512;
    static const unsigned out_height = 9;
    static const unsigned out_width = 9;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};










// q_conv2d_batchnorm_19
struct config67_mult : nnet::dense_config {
    static const unsigned n_in = 4608/2;    //! 
    static const unsigned n_out = 512;
    static const unsigned reuse_factor = 2304;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 721454;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_19_accum_t accum_t;
    typedef bias67_t bias_t;
    typedef weight67_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config67 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 9;
    static const unsigned in_width = 9;
    static const unsigned n_chan = 512/2;   //!
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 512;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 7;
    static const unsigned out_width = 7;
    static const unsigned reuse_factor = 2304;
    static const unsigned n_zeros = 721454;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 49;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_19_accum_t accum_t;
    typedef bias67_t bias_t;
    typedef weight67_t weight_t;
    typedef config67_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config67::filt_height * config67::filt_width> config67::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};



struct config67_mult_1 : nnet::dense_config {
    static const unsigned n_in = 4608/2;    //! 
    static const unsigned n_out = 512;
    static const unsigned reuse_factor = 2304;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 721454;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_19_accum_t accum_t;
    typedef bias67_t bias_t;
    typedef weight67_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config67_1 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 9;
    static const unsigned in_width = 9;
    static const unsigned n_chan = 512/2;   //!
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 512;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 7;
    static const unsigned out_width = 7;
    static const unsigned reuse_factor = 2304;
    static const unsigned n_zeros = 721454;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 49;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_19_accum_t accum_t;
    typedef bias67_t bias_t;
    typedef weight67_t weight_t;
    typedef config67_mult_1 mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};


struct config67_mult_2 : nnet::dense_config {
    static const unsigned n_in = 4608/2;    //! 
    static const unsigned n_out = 512;
    static const unsigned reuse_factor = 2304;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 721454;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv2d_batchnorm_19_accum_t accum_t;
    typedef bias67_t bias_t;
    typedef weight67_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config67_2 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 9;
    static const unsigned in_width = 9;
    static const unsigned n_chan = 512/2;   //!
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 512;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 7;
    static const unsigned out_width = 7;
    static const unsigned reuse_factor = 2304;
    static const unsigned n_zeros = 721454;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 49;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef q_conv2d_batchnorm_19_accum_t accum_t;
    typedef bias67_t bias_t;
    typedef weight67_t weight_t;
    typedef config67_mult_2 mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};


struct config67_split {
    static const unsigned in_height = 9;
    static const unsigned in_width = 9;
    static const unsigned n_chan = 512/2;   //!
};



struct config67_add {
    static const unsigned n_filt = 512;
    static const unsigned out_height = 7;
    static const unsigned out_width = 7;
};









// activation_19
struct relu_config69 : nnet::activ_config {
    static const unsigned n_in = 25088;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef activation_19_table_t table_t;
};

// add_7
struct config70 : nnet::merge_config {
    static const unsigned n_elem = OUT_HEIGHT_57*OUT_WIDTH_57*N_FILT_57;
};

// global_average_pooling2d
struct config71 : nnet::pooling2d_config {
    static const unsigned in_height = 7;
    static const unsigned in_width = 7;
    static const unsigned n_filt = 512;
    static const nnet::Pool_Op pool_op = nnet::Average;
    static const unsigned reuse_factor = 1;
    typedef global_average_pooling2d_accum_t accum_t;
};

// q_dense
struct config72 : nnet::dense_config {
    static const unsigned n_in = 512;
    static const unsigned n_out = 1000;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 128000;
    static const unsigned n_zeros = 63595;
    static const unsigned n_nonzeros = 448405;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef q_dense_accum_t accum_t;
    typedef bias72_t bias_t;
    typedef weight72_t weight_t;
    typedef layer72_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// activation_20
struct softmax_config74 : nnet::activ_config {
    static const unsigned n_in = 1000;
    static const unsigned table_size = 256;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 100;
    static const unsigned axis = -1;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::stable;
    typedef activation_20_exp_table_t exp_table_t;
    typedef activation_20_inv_table_t inv_table_t;
};


#endif
