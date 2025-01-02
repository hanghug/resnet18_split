#ifndef NNET_POOLING_STREAM_H_
#define NNET_POOLING_STREAM_H_

#include "ap_shift_reg.h"
#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_conv_stream.h"
#include "nnet_pooling.h"
#include "utils/x_hls_utils.h"

namespace nnet {

// *************************************************
//       Max/average pooling
// *************************************************

template <class T, int N, class CONFIG_T> T reduce_pool(T x[N]) {
    #pragma HLS INLINE
    if (CONFIG_T::pool_op == Max) {
        Op_max<T> op_max;
        return reduce<T, N, Op_max<T>>(x, op_max);
    } else {
        Op_add<T> op_add;
        T sum = reduce<T, N, Op_add<T>>(x, op_add);
        return sum / N;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void compute_pool_buffer_2d(const data_T &in_elem,
                            ap_shift_reg<typename data_T::value_type, CONFIG_T::in_width>
                                line_buffer[MAX(CONFIG_T::pool_height - 1, 1)][CONFIG_T::n_filt],
                            hls::stream<res_T> &res) {
    #pragma HLS INLINE
    const static int lShiftX = CONFIG_T::pool_width - 1;
    const static int lShiftY = CONFIG_T::pool_height - 1;
    static int pX = 0; // pixel X
    static int pY = 0; // pixel Y
    static int sX = 0; // stride X
    static int sY = 0; // stride Y

    typename CONFIG_T::accum_t pool_window[CONFIG_T::pool_height * CONFIG_T::pool_width];
    #pragma HLS ARRAY_PARTITION variable=pool_window complete

    static typename data_T::value_type kernel_data[CONFIG_T::pool_height * CONFIG_T::pool_width * CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable = kernel_data complete dim = 0

    res_T res_pack;
    PRAGMA_DATA_PACK(res_pack)

    // Add pixel into line buffer, return pooling kernels
    nnet::shift_line_buffer<data_T, CONFIG_T>(in_elem, line_buffer, kernel_data);

    // Can compute pooling output
    if ((sX - lShiftX) == 0 && (sY - lShiftY) == 0 && pY > lShiftY - 1 && pX > lShiftX - 1) {
    FiltLoop:
        for (unsigned i_ic = 0; i_ic < CONFIG_T::n_filt; i_ic++) {
        #pragma HLS PIPELINE

        // Retrieve data for current channel
        PoolLoop:
            for (unsigned i_ihw = 0; i_ihw < CONFIG_T::pool_height * CONFIG_T::pool_width; i_ihw++) {
                pool_window[i_ihw] = kernel_data[i_ihw * CONFIG_T::n_filt + i_ic];
            }

            // Compute Pooling
            res_pack[i_ic] =
                reduce_pool<typename CONFIG_T::accum_t, CONFIG_T::pool_height * CONFIG_T::pool_width, CONFIG_T>(pool_window);
        }

        // Write to output
        res.write(res_pack);
    }

    // Counter Housekeeping
    if (pX + 1 == CONFIG_T::in_width) // Includes padding, end of line (padded)
    {
        pX = 0;
        sX = 0;
        if (pY + 1 == CONFIG_T::in_height) { // Reached bottom of image
            pY = 0;
            sY = 0;
        } else { // Next line
            pY = pY + 1;
            // Update stride (threshold) ? subtract stride : increment stride
            sY = ((sY - lShiftY) == 0) ? sY - CONFIG_T::stride_height + 1 : sY + 1;
        }
    } else {
        pX = pX + 1;
        // Update stride (threshold) ? subtract stride : increment stride
        sX = ((sX - lShiftX) == 0) ? sX - CONFIG_T::stride_width + 1 : sX + 1;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void pooling2d_cl(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    assert(CONFIG_T::implementation == conv_implementation::linebuffer &&
           "Only \"linebuffer\" implementation is supported in Vitis HLS.");

    #pragma HLS INLINE recursive
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::pool_height == CONFIG_T::stride_height && CONFIG_T::pool_width == CONFIG_T::stride_width);

    static ap_shift_reg<typename data_T::value_type, CONFIG_T::in_width> line_buffer[MAX(CONFIG_T::pool_height - 1, 1)]
                                                                                    [CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable = line_buffer complete dim = 2

ReadInputHeight:
    for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
    ReadInputWidth:
        for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
            #pragma HLS LOOP_FLATTEN
            #pragma HLS PIPELINE

            compute_pool_buffer_2d<data_T, res_T, CONFIG_T>(data.read(), line_buffer, res);
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void compute_pool_buffer_2d(hls::stream<data_T> (&data)[CONFIG_T::n_chan],
                            ap_shift_reg<data_T, CONFIG_T::in_width>
                                line_buffer[MAX(CONFIG_T::pool_height - 1, 1)][CONFIG_T::n_filt],
                            hls::stream<res_T> (&res)[CONFIG_T::n_filt]) {
    #pragma HLS INLINE
    const static int lShiftX = CONFIG_T::pool_width - 1;
    const static int lShiftY = CONFIG_T::pool_height - 1;
    static int pX = 0; // pixel X
    static int pY = 0; // pixel Y
    static int sX = 0; // stride X
    static int sY = 0; // stride Y

    typename CONFIG_T::accum_t pool_window[CONFIG_T::pool_height * CONFIG_T::pool_width];
    #pragma HLS ARRAY_PARTITION variable=pool_window complete

    static data_T kernel_data[CONFIG_T::pool_height * CONFIG_T::pool_width * CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable = kernel_data complete dim = 0

    res_T res_pack[CONFIG_T::n_filt];

    // Add pixel into line buffer, return pooling kernels
    nnet::shift_line_buffer<data_T, CONFIG_T>(data, line_buffer, kernel_data);

    // Can compute pooling output
    if ((sX - lShiftX) == 0 && (sY - lShiftY) == 0 && pY > lShiftY - 1 && pX > lShiftX - 1) {
    FiltLoop:
        for (unsigned i_ic = 0; i_ic < CONFIG_T::n_filt; i_ic++) {
        #pragma HLS PIPELINE

        // Retrieve data for current channel
        PoolLoop:
            for (unsigned i_ihw = 0; i_ihw < CONFIG_T::pool_height * CONFIG_T::pool_width; i_ihw++) {
                pool_window[i_ihw] = kernel_data[i_ihw * CONFIG_T::n_filt + i_ic];
            }

            // Compute Pooling
            res_pack[i_ic] =
                reduce_pool<typename CONFIG_T::accum_t, CONFIG_T::pool_height * CONFIG_T::pool_width, CONFIG_T>(pool_window);
        }

        // Write to output
        for (unsigned i_ic = 0; i_ic < CONFIG_T::n_filt; i_ic++) {
            res[i_ic].write(res_pack[i_ic]);
        }
    }

    // Counter Housekeeping
    if (pX + 1 == CONFIG_T::in_width) // Includes padding, end of line (padded)
    {
        pX = 0;
        sX = 0;
        if (pY + 1 == CONFIG_T::in_height) { // Reached bottom of image
            pY = 0;
            sY = 0;
        } else { // Next line
            pY = pY + 1;
            // Update stride (threshold) ? subtract stride : increment stride
            sY = ((sY - lShiftY) == 0) ? sY - CONFIG_T::stride_height + 1 : sY + 1;
        }
    } else {
        pX = pX + 1;
        // Update stride (threshold) ? subtract stride : increment stride
        sX = ((sX - lShiftX) == 0) ? sX - CONFIG_T::stride_width + 1 : sX + 1;
    }
}


template <class data_T, class res_T, typename CONFIG_T>
void compute_pool_buffer_2d(hls::stream<data_T> (&data)[CONFIG_T::n_chan/data_T::size],
                            ap_shift_reg<typename data_T::value_type, CONFIG_T::in_width>
                                line_buffer[MAX(CONFIG_T::pool_height - 1, 1)][CONFIG_T::n_filt],
                            hls::stream<res_T> (&res)[CONFIG_T::n_filt/res_T::size]) {
    #pragma HLS INLINE
    const static int lShiftX = CONFIG_T::pool_width - 1;
    const static int lShiftY = CONFIG_T::pool_height - 1;
    static int pX = 0; // pixel X
    static int pY = 0; // pixel Y
    static int sX = 0; // stride X
    static int sY = 0; // stride Y

    typename CONFIG_T::accum_t pool_window[CONFIG_T::pool_height * CONFIG_T::pool_width];
    #pragma HLS ARRAY_PARTITION variable=pool_window complete

    static typename data_T::value_type kernel_data[CONFIG_T::pool_height * CONFIG_T::pool_width * CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable = kernel_data complete dim = 0

    typename res_T::value_type res_array[CONFIG_T::n_filt];

    // Add pixel into line buffer, return pooling kernels
    nnet::shift_line_buffer<data_T, CONFIG_T>(data, line_buffer, kernel_data);

    // Can compute pooling output
    if ((sX - lShiftX) == 0 && (sY - lShiftY) == 0 && pY > lShiftY - 1 && pX > lShiftX - 1) {
    FiltLoop:
        for (unsigned i_ic = 0; i_ic < CONFIG_T::n_filt; i_ic++) {
        #pragma HLS PIPELINE

        // Retrieve data for current channel
        PoolLoop:
            for (unsigned i_ihw = 0; i_ihw < CONFIG_T::pool_height * CONFIG_T::pool_width; i_ihw++) {
                pool_window[i_ihw] = kernel_data[i_ihw * CONFIG_T::n_filt + i_ic];
            }

            // Compute Pooling
            res_array[i_ic] =
                reduce_pool<typename CONFIG_T::accum_t, CONFIG_T::pool_height * CONFIG_T::pool_width, CONFIG_T>(pool_window);
        }

        // Write to output
        for (unsigned j = 0; j < CONFIG_T::n_filt/res_T::size; j++) {
            #pragma HLS UNROLL
            res_T res_pack;
            for (unsigned k = 0; k < res_T::size; k++){
                res_pack[k] = res_array[j * res_T::size + k];
            }
            res[j].write(res_pack);
        }
    }

    // Counter Housekeeping
    if (pX + 1 == CONFIG_T::in_width) // Includes padding, end of line (padded)
    {
        pX = 0;
        sX = 0;
        if (pY + 1 == CONFIG_T::in_height) { // Reached bottom of image
            pY = 0;
            sY = 0;
        } else { // Next line
            pY = pY + 1;
            // Update stride (threshold) ? subtract stride : increment stride
            sY = ((sY - lShiftY) == 0) ? sY - CONFIG_T::stride_height + 1 : sY + 1;
        }
    } else {
        pX = pX + 1;
        // Update stride (threshold) ? subtract stride : increment stride
        sX = ((sX - lShiftX) == 0) ? sX - CONFIG_T::stride_width + 1 : sX + 1;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void pooling2d_cl(hls::stream<data_T> (&data)[CONFIG_T::n_chan], hls::stream<res_T> (&res)[CONFIG_T::n_filt]) {
    assert(CONFIG_T::implementation == conv_implementation::linebuffer &&
           "Only \"linebuffer\" implementation is supported in Vitis HLS.");

    #pragma HLS INLINE recursive
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::pool_height == CONFIG_T::stride_height && CONFIG_T::pool_width == CONFIG_T::stride_width);

    static ap_shift_reg<data_T, CONFIG_T::in_width> line_buffer[MAX(CONFIG_T::pool_height - 1, 1)]
                                                                                    [CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable = line_buffer complete dim = 2

ReadInputHeight:
    for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
    ReadInputWidth:
        for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
            #pragma HLS LOOP_FLATTEN
            #pragma HLS PIPELINE

            compute_pool_buffer_2d<data_T, res_T, CONFIG_T>(data, line_buffer, res);
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void pooling2d_cl(hls::stream<data_T> (&data)[CONFIG_T::n_chan/data_T::size], hls::stream<res_T> (&res)[CONFIG_T::n_filt/res_T::size]) {
    assert(CONFIG_T::implementation == conv_implementation::linebuffer &&
           "Only \"linebuffer\" implementation is supported in Vitis HLS.");

    #pragma HLS INLINE recursive
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::pool_height == CONFIG_T::stride_height && CONFIG_T::pool_width == CONFIG_T::stride_width);

    static ap_shift_reg<typename data_T::value_type, CONFIG_T::in_width> line_buffer[MAX(CONFIG_T::pool_height - 1, 1)]
                                                                                    [CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable = line_buffer complete dim = 2

ReadInputHeight:
    for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
    ReadInputWidth:
        for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
            #pragma HLS LOOP_FLATTEN
            #pragma HLS PIPELINE

            compute_pool_buffer_2d<data_T, res_T, CONFIG_T>(data, line_buffer, res);
        }
    }
}

// *************************************************
//                  Pooling 1D
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void compute_pool_buffer_1d(const data_T &in_elem, hls::stream<res_T> &res) {
    #pragma HLS INLINE
    const static int lShiftX = CONFIG_T::pool_width - 1;
    // Counters
    static int pX = 0;
    static int sX = 0;

    typename CONFIG_T::accum_t pool_window[CONFIG_T::pool_width];
    #pragma HLS ARRAY_PARTITION variable=pool_window complete

    static typename data_T::value_type kernel_data[CONFIG_T::pool_width * CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable = kernel_data complete dim = 0

    res_T res_pack;
    PRAGMA_DATA_PACK(res_pack)

    // Add pixel into line buffer, return pooling kernels
    // 1D case line buffer not necessary. Put directly into the kernel_data buffer
    nnet::kernel_shift_1d<data_T, CONFIG_T>(in_elem, kernel_data);

    // Can compute pooling output
    if ((sX - lShiftX) == 0 && pX > lShiftX - 1) {
    FiltLoop:
        for (unsigned i_ic = 0; i_ic < CONFIG_T::n_filt; i_ic++) {
        #pragma HLS PIPELINE

        // Retrieve data for current channel
        PoolLoop:
            for (unsigned i_iw = 0; i_iw < CONFIG_T::pool_width; i_iw++) {
                pool_window[i_iw] = kernel_data[i_iw * CONFIG_T::n_filt + i_ic];
            }

            // Compute Pooling
            res_pack[i_ic] = reduce_pool<typename CONFIG_T::accum_t, CONFIG_T::pool_width, CONFIG_T>(pool_window);
        }

        // Write to output
        res.write(res_pack);
    }

    // Counter Housekeeping
    if (pX + 1 == CONFIG_T::n_in) // Includes padding, end of line (padded)
    {
        pX = 0;
        sX = 0;
    } else {
        pX = pX + 1;
        // Update stride (threshold) ? subtract stride : increment stride
        sX = ((sX - lShiftX) == 0) ? sX - CONFIG_T::stride_width + 1 : sX + 1;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void pooling1d_cl(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    assert(CONFIG_T::implementation == conv_implementation::linebuffer &&
           "Only \"linebuffer\" implementation is supported in Vitis HLS.");
    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);

    #pragma HLS inline recursive

ReadInputWidth:
    for (unsigned i_iw = 0; i_iw < CONFIG_T::n_in; i_iw++) {
        #pragma HLS PIPELINE
        compute_pool_buffer_1d<data_T, res_T, CONFIG_T>(data.read(), res);
    }
}

// *************************************************
//       Global max/average pooling
// *************************************************

template <class T, int N, class CONFIG_T> T reduce_global_pool(T x, T y[N]) {
    #pragma HLS INLINE
    if (CONFIG_T::pool_op == Max) {
        Op_max<T> op_max;
        T y_max = reduce<T, N, Op_max<T>>(y, op_max);
        return (x > y_max) ? x : y_max;
    } else {
        Op_add<T> op_add;
        T y_sum = reduce<T, N, Op_add<T>>(y, op_add);
        return x + y_sum;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void compute_global_pool(const data_T &in_elem, typename CONFIG_T::accum_t data_window[CONFIG_T::n_filt]) {
PoolFilt:
    for (unsigned c = 0; c < CONFIG_T::n_filt; c++) {
        #pragma HLS UNROLL

        typename CONFIG_T::accum_t data_pack[data_T::size / CONFIG_T::n_filt];
        #pragma HLS ARRAY_PARTITION variable=data_pack complete dim=0

    PixelLoop:
        for (unsigned p = 0; p < data_T::size / CONFIG_T::n_filt; p++) {
            #pragma HLS UNROLL
            data_pack[p] = in_elem[p * CONFIG_T::n_filt + c];
        }
        data_window[c] = reduce_global_pool<typename CONFIG_T::accum_t, data_T::size / CONFIG_T::n_filt, CONFIG_T>(
            data_window[c], data_pack);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void global_pooling2d_cl(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::pool_height == CONFIG_T::stride_height && CONFIG_T::pool_width == CONFIG_T::stride_width);

    constexpr int total_pixels = CONFIG_T::in_height * CONFIG_T::in_width;  //!
    
    typename CONFIG_T::accum_t data_window[CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable=data_window complete

    typename CONFIG_T::accum_t init = 0;
    if (CONFIG_T::pool_op == Max) {
        init = hls::numeric_limits<typename CONFIG_T::accum_t>::min();
    }

PoolInitLoop:
    for (unsigned i_init = 0; i_init < CONFIG_T::n_filt; i_init++) {
        #pragma HLS UNROLL
        data_window[i_init] = init;
    }

ReadInputHeight:
    for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
    ReadInputWidth:
        for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width / (data_T::size / CONFIG_T::n_filt); i_iw++) {
            #pragma HLS LOOP_FLATTEN
            compute_global_pool<data_T, res_T, CONFIG_T>(data.read(), data_window);
        }
    }

    if (CONFIG_T::pool_op == Max) {
    MaxPoolRes:
        for (unsigned i_res = 0; i_res < CONFIG_T::n_filt / res_T::size; i_res++) {
            #pragma HLS PIPELINE

            res_T res_pack;
            PRAGMA_DATA_PACK(res_pack)
        MaxPoolPack:
            for (unsigned i_pack = 0; i_pack < res_T::size; i_pack++) {
                #pragma HLS UNROLL
                res_pack[i_pack] = data_window[i_pack];
            }
            res.write(res_pack);
        }
    } else {
    AvgPoolRes:
        for (unsigned i_res = 0; i_res < CONFIG_T::n_filt / res_T::size; i_res++) {
            //#pragma HLS PIPELINE 

            res_T res_pack;
            PRAGMA_DATA_PACK(res_pack)
        AvgPoolPack:
            for (unsigned i_pack = 0; i_pack < res_T::size; i_pack++) {
                #pragma HLS PIPELINE off
                #pragma HLS UNROLL off   //!避免浪费过的dsp，下面一行就要用4个dsp，如果展开就要用res_T::size*4个dsp。
                res_pack[i_pack] = data_window[i_pack] / total_pixels;
            }
            res.write(res_pack);
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void compute_global_pool(hls::stream<data_T> (&data)[CONFIG_T::n_filt], typename CONFIG_T::accum_t data_window[CONFIG_T::n_filt]) {
PoolFilt:
    for (unsigned c = 0; c < CONFIG_T::n_filt; c++) {
        #pragma HLS UNROLL

        typename CONFIG_T::accum_t data_pack;
        #pragma HLS ARRAY_PARTITION variable=data_pack complete dim=0

        data_pack = data[c].read();
        data_window[c] = reduce_global_pool<typename CONFIG_T::accum_t, 1, CONFIG_T>(
            data_window[c], &data_pack);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void compute_global_pool(hls::stream<data_T> (&data)[CONFIG_T::n_filt/data_T::size], typename CONFIG_T::accum_t data_window[CONFIG_T::n_filt]) {
PoolFilt:
    for (unsigned j = 0; j < CONFIG_T::n_filt/data_T::size; j++) {
        #pragma HLS UNROLL
        data_T data_pack = data[j].read();
        // #pragma HLS ARRAY_PARTITION variable=data_accum complete dim=0
        for (unsigned k = 0; k < data_T::size; k++) {
            #pragma HLS UNROLL
            typename CONFIG_T::accum_t data_accum = data_pack[k];
            data_window[j*data_T::size + k] = reduce_global_pool<typename CONFIG_T::accum_t, 1, CONFIG_T>(
                                        data_window[j*data_T::size + k], &data_accum);
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void global_pooling2d_cl(hls::stream<data_T> (&data)[CONFIG_T::n_filt], hls::stream<res_T> (&res)[CONFIG_T::n_filt]) {
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::pool_height == CONFIG_T::stride_height && CONFIG_T::pool_width == CONFIG_T::stride_width);

    constexpr int total_pixels = CONFIG_T::in_height * CONFIG_T::in_width;  //!
    
    typename CONFIG_T::accum_t data_window[CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable=data_window complete

    typename CONFIG_T::accum_t init = 0;
    if (CONFIG_T::pool_op == Max) {
        init = hls::numeric_limits<typename CONFIG_T::accum_t>::min();
    }

PoolInitLoop:
    for (unsigned i_init = 0; i_init < CONFIG_T::n_filt; i_init++) {
        #pragma HLS UNROLL
        data_window[i_init] = init;
    }

ReadInputHeight:
    for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
    ReadInputWidth:
        for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
            #pragma HLS LOOP_FLATTEN
            compute_global_pool<data_T, res_T, CONFIG_T>(data, data_window);
        }
    }

    if (CONFIG_T::pool_op == Max) {
    MaxPoolPack:
        for (unsigned i_pack = 0; i_pack < CONFIG_T::n_filt; i_pack++) {
            #pragma HLS UNROLL
            res[i_pack].write(data_window[i_pack]);
        }
    } else {
        res_T res_pack[CONFIG_T::n_filt];
    AvgPoolPack:
        for (unsigned i_pack = 0; i_pack < CONFIG_T::n_filt; i_pack++) {
            #pragma HLS PIPELINE off
            #pragma HLS UNROLL off   //!避免浪费过的dsp，下面一行就要用4个dsp，如果展开就要用res_T::size*4个dsp。
            res_pack[i_pack] = data_window[i_pack] / total_pixels;
        }
    WriteLoop:
        for (unsigned i_pack = 0; i_pack < CONFIG_T::n_filt; i_pack++) {
            #pragma HLS UNROLL
            res[i_pack].write(res_pack[i_pack]);
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void global_pooling2d_cl(hls::stream<data_T> (&data)[CONFIG_T::n_filt/data_T::size], hls::stream<res_T> (&res)[CONFIG_T::n_filt/res_T::size]) {
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::pool_height == CONFIG_T::stride_height && CONFIG_T::pool_width == CONFIG_T::stride_width);

    constexpr int total_pixels = CONFIG_T::in_height * CONFIG_T::in_width;  //!
    
    typename CONFIG_T::accum_t data_window[CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable=data_window complete

    typename CONFIG_T::accum_t init = 0;
    if (CONFIG_T::pool_op == Max) {
        init = hls::numeric_limits<typename CONFIG_T::accum_t>::min();
    }

PoolInitLoop:
    for (unsigned i_init = 0; i_init < CONFIG_T::n_filt; i_init++) {
        #pragma HLS UNROLL
        data_window[i_init] = init;
    }

ReadInputHeight:
    for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
    ReadInputWidth:
        for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
            #pragma HLS LOOP_FLATTEN
            compute_global_pool<data_T, res_T, CONFIG_T>(data, data_window);
        }
    }

    if (CONFIG_T::pool_op == Max) {
    MaxPoolPack:
        for (unsigned j = 0; j < CONFIG_T::n_filt/res_T::size; j++) {
            #pragma HLS UNROLL
            res_T res_pack;
            for (unsigned k = 0; k < res_T::size; k++) {
                #pragma HLS UNROLL
                res_pack[k] = data_window[j*res_T::size+k];
            }
            res[j].write(res_pack);
        }
    } else {
        typename res_T::value_type res_array[CONFIG_T::n_filt];
    AvgPoolPack:
        for (unsigned c = 0; c < CONFIG_T::n_filt; c++) {
            #pragma HLS PIPELINE off
            #pragma HLS UNROLL off   //!避免浪费过的dsp，下面一行就要用4个dsp，如果展开就要用res_T::size*4个dsp。
            res_array[c] = data_window[c] / total_pixels;
        }
    WriteLoop:
        for (unsigned j = 0; j < CONFIG_T::n_filt/res_T::size; j++) {
            #pragma HLS UNROLL
            res_T res_pack;
            for (unsigned k = 0; k < res_T::size; k++) {
                #pragma HLS UNROLL
                res_pack[k] = res_array[j*res_T::size+k];
            }
            res[j].write(res_pack);
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void global_pooling1d_cl(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::pool_width == CONFIG_T::stride_width);

    typename CONFIG_T::accum_t data_window[CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable=data_window complete

    typename CONFIG_T::accum_t init = 0;
    if (CONFIG_T::pool_op == Max) {
        init = hls::numeric_limits<typename CONFIG_T::accum_t>::min();
    }

PoolInitLoop:
    for (unsigned i_init = 0; i_init < CONFIG_T::n_filt; i_init++) {
        #pragma HLS UNROLL
        data_window[i_init] = init;
    }

ReadInput:
    for (unsigned i_iw = 0; i_iw < CONFIG_T::n_in / (data_T::size / CONFIG_T::n_filt); i_iw++) {
        #pragma HLS LOOP_FLATTEN
        compute_global_pool<data_T, res_T, CONFIG_T>(data.read(), data_window);
    }

    if (CONFIG_T::pool_op == Max) {
    MaxPoolRes:
        for (unsigned i_res = 0; i_res < CONFIG_T::n_filt / res_T::size; i_res++) {
            #pragma HLS PIPELINE

            res_T res_pack;
            PRAGMA_DATA_PACK(res_pack)
        MaxPoolPack:
            for (unsigned i_pack = 0; i_pack < res_T::size; i_pack++) {
                #pragma HLS UNROLL
                res_pack[i_pack] = data_window[i_pack];
            }
            res.write(res_pack);
        }
    } else {
    AvgPoolRes:
        for (unsigned i_res = 0; i_res < CONFIG_T::n_filt / res_T::size; i_res++) {
            #pragma HLS PIPELINE

            res_T res_pack;
            PRAGMA_DATA_PACK(res_pack)
        AvgPoolPack:
            for (unsigned i_pack = 0; i_pack < res_T::size; i_pack++) {
                #pragma HLS UNROLL
                res_pack[i_pack] = data_window[i_pack] / CONFIG_T::n_in;
            }
            res.write(res_pack);
        }
    }
}

} // namespace nnet

#endif
