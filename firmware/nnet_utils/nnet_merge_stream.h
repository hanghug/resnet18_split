#ifndef NNET_MERGE_STREAM_H_
#define NNET_MERGE_STREAM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include <math.h>

namespace nnet {

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void add(hls::stream<input1_T> &data1, hls::stream<input2_T> &data2, hls::stream<res_T> &res) {
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

AddLoop:
    for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        #pragma HLS PIPELINE

        input1_T in_data1 = data1.read();
        input2_T in_data2 = data2.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    AddPack:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            out_data[j] = in_data1[j] + in_data2[j];
        }

        res.write(out_data);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T, int n_chan>
void add(hls::stream<input1_T> (&data1)[n_chan], hls::stream<input2_T> (&data2)[n_chan], hls::stream<res_T> (&res)[n_chan]) {

AddLoop:
    for (int i = 0; i < CONFIG_T::n_elem / n_chan; i++) {
        #pragma HLS PIPELINE

        input1_T in_data1[n_chan];
        input2_T in_data2[n_chan];
        res_T out_data[n_chan];
    
    ReadPack:
        for (int j = 0; j < n_chan; j++) {
            #pragma HLS UNROLL
            in_data1[j] = data1[j].read();
            in_data2[j] = data2[j].read();
        }

    AddPack:
        for (int j = 0; j < n_chan; j++) {
            #pragma HLS UNROLL
            out_data[j] = in_data1[j] + in_data2[j];
        }

    WritePack:
        for (int j = 0; j < n_chan; j++) {
            #pragma HLS UNROLL
            res[j].write(out_data[j]);
        }
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T, int n_chan>
void add(hls::stream<input1_T> (&data1)[n_chan/input1_T::size], hls::stream<input2_T> (&data2)[n_chan/input2_T::size], hls::stream<res_T> (&res)[n_chan/res_T::size]) {

AddLoop:
    for (int i = 0; i < CONFIG_T::n_elem / n_chan; i++) {
        #pragma HLS PIPELINE

        typename input1_T::value_type in_data1[n_chan];
        typename input2_T::value_type in_data2[n_chan];
        typename res_T::value_type out_data[n_chan];
    
    ReadPack1:
        for (int j = 0; j < n_chan/input1_T::size; j++) {
            #pragma HLS UNROLL
            input1_T in_data1_pack = data1[j].read();
            for (int k = 0; k < input1_T::size; k++) {
                #pragma HLS UNROLL
                in_data1[j*input1_T::size + k] = in_data1_pack[k];
            }
        }
    ReadPack2:
        for (int j = 0; j < n_chan/input2_T::size; j++) {
            #pragma HLS UNROLL
            input2_T in_data2_pack = data2[j].read();
            for (int k = 0; k < input2_T::size; k++) {
                #pragma HLS UNROLL
                in_data2[j*input2_T::size + k] = in_data2_pack[k];
            }
        }


    Add:
        for (int j = 0; j < n_chan; j++) {
            #pragma HLS UNROLL
            out_data[j] = in_data1[j] + in_data2[j];
        }

    WritePack:
        for (int j = 0; j < n_chan/res_T::size; j++) {
            #pragma HLS UNROLL
            res_T out_data_pack;
            for (int k = 0; k < res_T::size; k++) {
                #pragma HLS UNROLL
                out_data_pack[k] = out_data[j*res_T::size + k];
            }
            res[j].write(out_data_pack);
        }
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void subtract(hls::stream<input1_T> &data1, hls::stream<input2_T> &data2, hls::stream<res_T> &res) {
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

SubtractLoop:
    for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        #pragma HLS PIPELINE

        input1_T in_data1 = data1.read();
        input2_T in_data2 = data2.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    SubtractPack:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            out_data[j] = in_data1[j] - in_data2[j];
        }

        res.write(out_data);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void multiply(hls::stream<input1_T> &data1, hls::stream<input2_T> &data2, hls::stream<res_T> &res) {
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

MultiplyLoop:
    for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        input1_T in_data1 = data1.read();
        input2_T in_data2 = data2.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    MultiplyPack:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            out_data[j] = in_data1[j] * in_data2[j];
        }

        res.write(out_data);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void average(hls::stream<input1_T> &data1, hls::stream<input2_T> &data2, hls::stream<res_T> &res) {
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

AverageLoop:
    for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        input1_T in_data1 = data1.read();
        input2_T in_data2 = data2.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    AveragePack:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            out_data[j] = (in_data1[j] + in_data2[j]) / (typename res_T::value_type)2;
        }

        res.write(out_data);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void maximum(hls::stream<input1_T> &data1, hls::stream<input2_T> &data2, hls::stream<res_T> &res) {
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

MaximumLoop:
    for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        input1_T in_data1 = data1.read();
        input2_T in_data2 = data2.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    MaximumPack:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            out_data[j] = (in_data1[j] > in_data2[j]) ? in_data1[j] : in_data2[j];
        }

        res.write(out_data);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void minimum(hls::stream<input1_T> &data1, hls::stream<input2_T> &data2, hls::stream<res_T> &res) {
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

MinimumLoop:
    for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        input1_T in_data1 = data1.read();
        input2_T in_data2 = data2.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    MinimumPack:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            out_data[j] = (in_data1[j] < in_data2[j]) ? in_data1[j] : in_data2[j];
        }

        res.write(out_data);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d_0(hls::stream<input1_T> &data1, hls::stream<input2_T> &data2, hls::stream<res_T> &res) {
ConcatLoopHeight1:
    for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
    ConcatLoopWidth1:
        for (int j = 0; j < CONFIG_T::n_elem1_1; j++) {
            #pragma HLS PIPELINE II=1

            input1_T in_data1 = data1.read();
            res_T out_data;
            PRAGMA_DATA_PACK(out_data)

        ConcatPackInput1:
            for (int k = 0; k < input1_T::size; k++) {
                #pragma HLS UNROLL
                out_data[k] = in_data1[k];
            }

            res.write(out_data);
        }
    }
ConcatLoopHeight2:
    for (int i = 0; i < CONFIG_T::n_elem2_0; i++) {
    ConcatLoopWidth2:
        for (int j = 0; j < CONFIG_T::n_elem2_1; j++) {
            #pragma HLS PIPELINE II=1

            input2_T in_data2 = data2.read();
            res_T out_data;
            PRAGMA_DATA_PACK(out_data)

        ConcatPackInput2:
            for (int k = 0; k < input2_T::size; k++) {
                #pragma HLS UNROLL
                out_data[k] = in_data2[k];
            }

            res.write(out_data);
        }
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d_1(hls::stream<input1_T> &data1, hls::stream<input2_T> &data2, hls::stream<res_T> &res) {
ConcatLoopHeight:
    for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
    ConcatLoopWidth1:
        for (int j = 0; j < CONFIG_T::n_elem1_1; j++) {
            #pragma HLS PIPELINE II=1

            input1_T in_data1 = data1.read();
            res_T out_data;
            PRAGMA_DATA_PACK(out_data)

        ConcatPackInput1:
            for (int k = 0; k < input1_T::size; k++) {
                #pragma HLS UNROLL
                out_data[k] = in_data1[k];
            }

            res.write(out_data);
        }
    ConcatLoopWidth2:
        for (int j = 0; j < CONFIG_T::n_elem2_1; j++) {
            #pragma HLS PIPELINE II=1

            input2_T in_data2 = data2.read();
            res_T out_data;
            PRAGMA_DATA_PACK(out_data)

        ConcatPackInput2:
            for (int k = 0; k < input2_T::size; k++) {
                #pragma HLS UNROLL
                out_data[k] = in_data2[k];
            }

            res.write(out_data);
        }
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d_2(hls::stream<input1_T> &data1, hls::stream<input2_T> &data2, hls::stream<res_T> &res) {
ConcatLoopHeight:
    for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
    ConcatLoopWidth:
        for (int j = 0; j < CONFIG_T::n_elem1_1; j++) {
            #pragma HLS PIPELINE II=1

            input1_T in_data1 = data1.read();
            input2_T in_data2 = data2.read();
            res_T out_data;
            PRAGMA_DATA_PACK(out_data)

        ConcatPackInput1:
            for (int k = 0; k < input1_T::size; k++) {
                #pragma HLS UNROLL
                out_data[k] = in_data1[k];
            }

        ConcatPackInput2:
            for (int k = 0; k < input2_T::size; k++) {
                #pragma HLS UNROLL
                out_data[input1_T::size + k] = in_data2[k];
            }

            res.write(out_data);
        }
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d(hls::stream<input1_T> &data1, hls::stream<input2_T> &data2, hls::stream<res_T> &res) {
    if (CONFIG_T::axis == 3 || CONFIG_T::axis == -1) {
        concatenate3d_2<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    } else if (CONFIG_T::axis == 2 || CONFIG_T::axis == -2) {
        concatenate3d_1<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    } else {
        concatenate3d_0<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate2d_0(hls::stream<input1_T> &data1, hls::stream<input2_T> &data2, hls::stream<res_T> &res) {
ConcatLoopHeight1:
    for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
        #pragma HLS PIPELINE II=1

        input1_T in_data1 = data1.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    ConcatPackInput1:
        for (int k = 0; k < input1_T::size; k++) {
            #pragma HLS UNROLL
            out_data[k] = in_data1[k];
        }

        res.write(out_data);
    }
ConcatLoopHeight2:
    for (int i = 0; i < CONFIG_T::n_elem2_0; i++) {
        #pragma HLS PIPELINE II=1

        input2_T in_data2 = data2.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    ConcatPackInput2:
        for (int k = 0; k < input2_T::size; k++) {
            #pragma HLS UNROLL
            out_data[k] = in_data2[k];
        }

        res.write(out_data);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate2d_1(hls::stream<input1_T> &data1, hls::stream<input2_T> &data2, hls::stream<res_T> &res) {
ConcatLoopHeight:
    for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
        #pragma HLS PIPELINE II=1

        input1_T in_data1 = data1.read();
        input2_T in_data2 = data2.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    ConcatPackInput1:
        for (int k = 0; k < input1_T::size; k++) {
            #pragma HLS UNROLL
            out_data[k] = in_data1[k];
        }

    ConcatPackInput2:
        for (int k = 0; k < input2_T::size; k++) {
            #pragma HLS UNROLL
            out_data[input1_T::size + k] = in_data2[k];
        }

        res.write(out_data);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate2d(hls::stream<input1_T> &data1, hls::stream<input2_T> &data2, hls::stream<res_T> &res) {
    if (CONFIG_T::axis == 2 || CONFIG_T::axis == -1) {
        concatenate2d_1<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    } else {
        concatenate2d_0<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate1d(hls::stream<input1_T> &data1, hls::stream<input2_T> &data2, hls::stream<res_T> &res) {
    res_T out_data;
    PRAGMA_DATA_PACK(out_data)
ConcatLoop1:
    for (int i = 0; i < CONFIG_T::n_elem1_0 / input1_T::size; i++) {
        #pragma HLS PIPELINE
        input1_T in_data1 = data1.read();
    ConcatPack1:
        for (int j = 0; j < input1_T::size; j++) {
            #pragma HLS UNROLL
            out_data[j + (i * input1_T::size)] = in_data1[j];
        }
    }
ConcatLoop2:
    for (int i = 0; i < CONFIG_T::n_elem2_0 / input2_T::size; i++) {
        #pragma HLS PIPELINE
        input2_T in_data2 = data2.read();
    ConcatPack2:
        for (int j = 0; j < input2_T::size; j++) {
            #pragma HLS UNROLL
            out_data[j + (i * input2_T::size) + (CONFIG_T::n_elem1_0)] = in_data2[j];
        }
    }
    res.write(out_data);
}
} // namespace nnet

#endif
