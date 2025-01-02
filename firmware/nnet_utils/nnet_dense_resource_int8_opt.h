//! 该版本中acc这个中间变量又加上了，希望有寄存器打拍？
#ifndef NNET_DENSE_RESOURCE_H_
#define NNET_DENSE_RESOURCE_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_mult.h"
#include <assert.h>
#include <math.h>

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void dense_resource_rf_leq_nin(
                               data_T data[CONFIG_T::n_in], 
                               res_T res[CONFIG_T::n_out],
                               typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                               typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

    const int rufactor = CONFIG_T::reuse_factor;
    const int multfactor = MIN(CONFIG_T::n_in, CONFIG_T::reuse_factor);
    const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, multfactor);
    const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, CONFIG_T::reuse_factor);
    const int multscale = multiplier_limit / CONFIG_T::n_out; //  == n_in / rf
    const int nin = CONFIG_T::n_in;
    const int nout = CONFIG_T::n_out;

    assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");
    assert((multiplier_limit == block_factor) && "This function is correct only for RF <= N_IN");
    assert((nout % 2 == 0) && "This function is correct only for N_OUT even");

    #pragma HLS function_instantiate variable=weights,biases
    //#pragma HLS RESOURCE variable=weights core=RAM_2P_BRAM Commenting out the deisgnation HLS seems to choose correctly
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete

    res_T acc[CONFIG_T::n_out];    //! 不用accumt_t的精度，因为这个一般不会配置，而引入这个acc的目的不是为了类型转换，是看是否能延迟几拍？
    #pragma HLS ARRAY_PARTITION variable=acc complete

InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];//!仔细确认CONFIG_T::accum_t和res_T的精度
    }

ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind

    MultLoop:
        for (int iout = 0; iout < nout; iout+=2){
            #pragma HLS UNROLL
            for (int im = 0; im < multscale; im++) {
                #pragma HLS UNROLL
                // (acc[iout], acc[iout+1]) = data[im*rufactor + ir] * (weights[ir + iout*nin + im*rufactor], weights[ir + (iout+1)*nin + im*rufactor]);
                
                // acc[iout] += static_cast<typename CONFIG_T::accum_t>(
                //     CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[im*rufactor + ir], weights[ir + iout*nin + im*rufactor]));
                // acc[iout+1] += static_cast<typename CONFIG_T::accum_t>(
                //     CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[im*rufactor + ir], weights[ir + (iout+1)*nin + im*rufactor]));
                ap_int<8> w1;
                ap_int<8> w2;
                ap_int<8> d;
                ap_int<27> ww;
                ap_int<34> r;
                ap_int<16> r1;
                ap_int<16> r2;
                ap_fixed<16,4> r1f;
                ap_fixed<16,4> r2f;
                w1.range() = weights[ir + iout*nin + im*rufactor].range();
                w2.range() = weights[ir + (iout+1)*nin + im*rufactor].range();
                d.range() = data[im*rufactor + ir].range();
                ww = (ap_int<27>)((ap_int<1>)0, w1, (ap_int<18>)0) + (ap_int<27>)w2;
                r = d * ww;
                r1.range() = r.range(33, 18);
                r2.range() = r.range(15, 0);
                r1 += r2[15];
                r1f.range() = r1.range();
                r2f.range() = r2.range();
                acc[iout] += static_cast<typename CONFIG_T::accum_t>(r1f);
                acc[iout+1] += static_cast<typename CONFIG_T::accum_t>(r2f);
            }
        }
    }

// Cast to "res_t" type
Result:
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        #pragma HLS UNROLL
        // res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
        res[ires] = acc[ires];
    }

}


template <class data_T, class res_T, typename CONFIG_T>
void dense_resource_rf_gt_nin_rem0(
                                   data_T data[CONFIG_T::n_in], 
                                   res_T res[CONFIG_T::n_out],
                                   typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                                   typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

    const int rufactor = CONFIG_T::reuse_factor;
    const int nin = CONFIG_T::n_in;
    const int nout = CONFIG_T::n_out;
    const int multiplier_limit = CONFIG_T::n_out;
    const int outscale = rufactor / CONFIG_T::n_in;
    const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, CONFIG_T::reuse_factor);

    assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");
    assert((rufactor > nin && rufactor % nin == 0) && "This function is correct only for RF > N_IN && RF % N_IN == 0");
    assert((block_factor % 2 == 0) && "This function is correct only for block_factor % 2 == 0");

    #pragma HLS function_instantiate variable=weights,biases
    #pragma HLS ARRAY_RESHAPE  variable=weights block factor=block_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete


    res_T acc[CONFIG_T::n_out];    //! 不用accumt_t的精度，因为这个一般不会配置，而引入这个acc的目的不是为了类型转换，是看是否能延迟几拍？
    #pragma HLS ARRAY_PARTITION variable=acc complete


InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }

ApplyReuseLoop:
    for (int i=0;i < outscale; i++) {
    ReuseLoop:
        for (int ir = 0; ir < nin; ir++) {
            #pragma HLS PIPELINE II=1 rewind

        MultLoop:
            for (int im = 0; im < block_factor; im+=2) {
                #pragma HLS UNROLL
                // acc[im*outscale + i] += static_cast<typename CONFIG_T::accum_t>(
                //     CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[ir], weights[ir + nin*i + im*rufactor]));
                // acc[(im+1)*outscale + i] += static_cast<typename CONFIG_T::accum_t>(
                //     CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[ir], weights[ir + nin*i + (im+1)*rufactor]));
                ap_int<8> w1;
                ap_int<8> w2;
                ap_int<8> d;
                ap_int<27> ww;
                ap_int<34> r;
                ap_int<16> r1;
                ap_int<16> r2;
                ap_fixed<16,4> r1f;
                ap_fixed<16,4> r2f;
                w1.range() = weights[ir + nin*i + im*rufactor].range();
                w2.range() = weights[ir + nin*i + (im+1)*rufactor].range();
                d.range() = data[ir].range();
                ww = (ap_int<27>)((ap_int<1>)0, w1, (ap_int<18>)0) + (ap_int<27>)w2;
                r = d * ww;
                r1.range() = r.range(33, 18);
                r2.range() = r.range(15, 0);
                r1 += r2[15];
                r1f.range() = r1.range();
                r2f.range() = r2.range();
                acc[im*outscale + i] += static_cast<typename CONFIG_T::accum_t>(r1f);
                acc[(im+1)*outscale + i] += static_cast<typename CONFIG_T::accum_t>(r2f);
            }
        }
    }

    // Cast to "res_t" type
Result:
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        #pragma HLS UNROLL
        // res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
        res[ires] = acc[ires];
    }

}



template <class data_T, class res_T, typename CONFIG_T>
void dense_resource(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                    typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                    typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

    #pragma HLS INLINE recursive

    if (CONFIG_T::reuse_factor <= CONFIG_T::n_in) {
        dense_resource_rf_leq_nin<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        dense_resource_rf_gt_nin_rem0<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } 
}

} // namespace nnet

#endif





//! 该版本中acc这个中间变量给省略了。
// #ifndef NNET_DENSE_RESOURCE_H_
// #define NNET_DENSE_RESOURCE_H_

// #include "hls_stream.h"
// #include "nnet_common.h"
// #include "nnet_mult.h"
// #include <assert.h>
// #include <math.h>

// namespace nnet {

// template <class data_T, class res_T, typename CONFIG_T>
// void dense_resource_rf_leq_nin(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
//                                typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
//                                typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

//     const int rufactor = CONFIG_T::reuse_factor;
//     const int multfactor = MIN(CONFIG_T::n_in, CONFIG_T::reuse_factor);
//     const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, multfactor);
//     const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, CONFIG_T::reuse_factor);
//     const int multscale = multiplier_limit / CONFIG_T::n_out;
//     const int nin = CONFIG_T::n_in;
//     const int nout = CONFIG_T::n_out;

//     assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");
//     assert((multiplier_limit == block_factor) && "This function is correct only for RF <= N_IN");

//     #pragma HLS function_instantiate variable=weights,biases
//     //#pragma HLS RESOURCE variable=weights core=RAM_2P_BRAM Commenting out the deisgnation HLS seems to choose correctly
//     #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
//     #pragma HLS ARRAY_PARTITION variable=biases complete

//     //typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
//     //#pragma HLS ARRAY_PARTITION variable=acc complete

// InitAccum:
//     for (int iacc = 0; iacc < nout; iacc++) {
//         #pragma HLS UNROLL
//         res[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
//     }

// ReuseLoop:
//     for (int ir = 0; ir < rufactor; ir++) {
//         #pragma HLS PIPELINE II=1 rewind

//         int w_index = ir;
//         int in_index = ir;
//         int out_index = 0;
//         int acc_step = 0;

//     MultLoop:
//         for (int im = 0; im < block_factor; im++) {
//             #pragma HLS UNROLL

//             res[out_index] += static_cast<typename CONFIG_T::accum_t>(
//                 CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index], weights[w_index]));

//             // Increment w_index
//             w_index += rufactor;
//             // Increment in_index
//             in_index += rufactor;
//             if (in_index >= nin) {
//                 in_index = ir;
//             }
//             // Increment out_index
//             if (acc_step + 1 >= multscale) {
//                 acc_step = 0;
//                 out_index++;
//             } else {
//                 acc_step++;
//             }
//         }
//     }

// // Cast to "res_t" type
// // Result:
// //     for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
// //         #pragma HLS UNROLL
// //         res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
// //     }
// }


// template <class data_T, class res_T, typename CONFIG_T>
// void dense_resource_rf_gt_nin_rem0(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
//                                    typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
//                                    typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

//     const int rufactor = CONFIG_T::reuse_factor;
//     const int nin = CONFIG_T::n_in;
//     const int nout = CONFIG_T::n_out;
//     const int multiplier_limit = CONFIG_T::n_out;
//     const int outscale = rufactor / CONFIG_T::n_in;
//     const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, CONFIG_T::reuse_factor);

//     assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");
//     assert((rufactor > nin && rufactor % nin == 0) && "This function is correct only for RF > N_IN && RF % N_IN == 0");

//     #pragma HLS function_instantiate variable=weights,biases
//     #pragma HLS ARRAY_RESHAPE  variable=weights block factor=block_factor
//     #pragma HLS ARRAY_PARTITION variable=biases complete


// InitAccum:
//     for (int iacc = 0; iacc < nout; iacc++) {
//         #pragma HLS UNROLL
//         res[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
//     }

// ApplyReuseLoop:
//     for (int i=0;i < outscale; i++) {
//     ReuseLoop:
//         for (int ir = 0; ir < nin; ir++) {
//             #pragma HLS PIPELINE II=1 rewind

//             int out_index = i;
//             int w_index = ir + nin * i;
//             int in_index = ir;

//         MultLoop:
//             for (int im = 0; im < block_factor; im++) {
//             #pragma HLS UNROLL
//             res[out_index] += static_cast<typename CONFIG_T::accum_t>(
//                 CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index], weights[w_index]));

//             w_index += rufactor;
//             out_index += outscale;
//             // if (w_index >= CONFIG_T::n_in * CONFIG_T::n_out)
//             //     break; // check out of bounds
//         }
//         }
//     }
// }

// template <class data_T, class res_T, typename CONFIG_T>
// void dense_resource(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
//                     typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
//                     typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

//     #pragma HLS INLINE recursive

//     if (CONFIG_T::reuse_factor <= CONFIG_T::n_in) {
//         dense_resource_rf_leq_nin<data_T, res_T, CONFIG_T>(data, res, weights, biases);
//     } else {
//         dense_resource_rf_gt_nin_rem0<data_T, res_T, CONFIG_T>(data, res, weights, biases);
//     } 
// }

// } // namespace nnet

// #endif







// template <class data_T, class res_T, typename CONFIG_T>
// void dense_resource_rf_gt_nin_rem0(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
//                                    typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
//                                    typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
//     //std::cout << "dog dog!!!!!?" << std::endl;
//     const int rufactor = CONFIG_T::reuse_factor;
//     const int nin = CONFIG_T::n_in;
//     const int nout = CONFIG_T::n_out;
//     const int multiplier_limit = CONFIG_T::n_out;
//     const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, CONFIG_T::reuse_factor);
//     //const int multscale = multiplier_limit / CONFIG_T::n_out;
//     const int outscale = rufactor / nin;
    

//     assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");
//     assert((rufactor > nin && rufactor % nin == 0) && "This function is correct only for RF > N_IN && RF % N_IN == 0");

//     #pragma HLS function_instantiate variable=weights,biases
//     #pragma HLS ARRAY_RESHAPE  variable=weights block factor=block_factor
//     #pragma HLS ARRAY_PARTITION variable=biases complete

//     //typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
//     //#pragma HLS ARRAY_PARTITION variable=acc complete

// InitAccum:
//     for (int iacc = 0; iacc < nout; iacc++) {
//         #pragma HLS UNROLL
//         res[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
//     }

 
// ReuseLoop1:
//     for (int ir = 0; ir < nin; ir++) {
//         #pragma HLS PIPELINE II=1 rewind

//         int w_index = ir;
//         int in_index = ir;
//         int out_index = 0;
        
//     MultLoop1:
//         for (int im = 0; im < block_factor; im++) {
//             #pragma HLS UNROLL
//             res[out_index] += static_cast<typename CONFIG_T::accum_t>(
//                 CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index], weights[w_index]));

//             w_index += rufactor;
//             // if (w_index >= CONFIG_T::n_in * CONFIG_T::n_out)
//             //     break; // check out of bounds
//             out_index += outscale;
//         }

//     }

// ReuseLoop2:
//     for (int ir = nin; ir < nin+nin; ir++) {
//         #pragma HLS PIPELINE II=1 rewind

//         int w_index = ir;
//         int in_index = ir-nin;
//         int out_index = 1;
        
//     MultLoop2:
//         for (int im = 0; im < block_factor; im++) {
//             #pragma HLS UNROLL
//             res[out_index] += static_cast<typename CONFIG_T::accum_t>(
//                 CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index], weights[w_index]));

//             w_index += rufactor;
//             // if (w_index >= CONFIG_T::n_in * CONFIG_T::n_out)
//             //     break; // check out of bounds
//             out_index += outscale;
//         }

//     }


// }