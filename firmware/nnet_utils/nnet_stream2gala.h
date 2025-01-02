#ifndef NNET_STREAM2GALA_H_
#define NNET_STREAM2GALA_H_

#include "hls_stream.h"
#include <iostream>

#ifdef CPU
#include "galapagos_interface.hpp"
#else
#include "galapagos_packet.h"
#endif

namespace nnet
{

template <int N_CHAN>
void galapagos_data_read(galapagos_interface *bridge_input, galapagos_packet (&arr_packet)[N_CHAN]) {
    #pragma HLS INLINE off

    for (unsigned j = 0; j < N_CHAN; j++) {
        #pragma HLS PIPELINE
        arr_packet[j] = bridge_input->read();
    }
}

template <class STREAM_TYPE, int TRANS_IN_PACKET, int N_CHAN, int data_len>
void galapagos_data_trans_chan(const galapagos_packet (&arr_packet)[N_CHAN], STREAM_TYPE (&arr_hls_data)[TRANS_IN_PACKET]) {
    #pragma HLS INLINE off
    GALAPGOS_DATA_TRANS_CHAN:
    for (unsigned j = 0; j < N_CHAN; j++) {
        #pragma HLS PIPELINE
        GALA_DATA_TRANS_IN_PACKET:
        for (int k = 0; k < TRANS_IN_PACKET; k++) {
            #pragma HLS UNROLL
            arr_hls_data[k][j].range() = arr_packet[j].data.range((k + 1) * data_len - 1, k * data_len);
        }
    }
}

template <class STREAM_TYPE, int TRANS_IN_PACKET, int LOOP_SIZE, int REMAIN_TRANS>
void hls_data_write(hls::stream<STREAM_TYPE> &hls_output, STREAM_TYPE (&arr_hls_data)[TRANS_IN_PACKET], unsigned i) {
    #pragma HLS INLINE off
    HLS_DATA_WRITE:
    for (unsigned j = 0; j < TRANS_IN_PACKET; j++) {
        #pragma HLS PIPELINE
        hls_output.write(arr_hls_data[j]);

        if (i == LOOP_SIZE - 1 && j == REMAIN_TRANS - 1) {   // 边界条件，易错
            break;
        }
    }
}

template <class STREAM_TYPE, int elements_in_stream, int data_len>
void galapagos_interface_2_hls_stream(
    galapagos_interface *bridge_input,
    hls::stream<STREAM_TYPE> &hls_output)
{

    #pragma HLS INLINE off

    const int N_CHAN = STREAM_TYPE::size;   //! 通道数，每一个hls_data[i]里包含的数的个数
    constexpr int TRANS_IN_PACKET = PACKET_DATA_LENGTH / data_len;  //! 一个packt可以存几个数，决定了并行读几次hls_data
    constexpr int LOOP_SIZE = nnet::ceil_div(elements_in_stream, TRANS_IN_PACKET);
    constexpr int REMAIN_TRANS = elements_in_stream - (LOOP_SIZE-1) * TRANS_IN_PACKET; //! REMAIN_TRANS一定小于TRANS_IN_PACKET

    galapagos_packet arr_packet[N_CHAN];
	#pragma HLS ARRAY_PARTITION variable=arr_packet complete

    STREAM_TYPE arr_hls_data[TRANS_IN_PACKET];
	#pragma HLS ARRAY_PARTITION variable=arr_hls_data complete


    GALAPAGOS_2_HLS_LOOP:
    for (unsigned i = 0; i < LOOP_SIZE; i++) {
        //#pragma HLS DATAFLOW
        #pragma HLS PIPELINE

        // 调用 galapagos_data_read
        galapagos_data_read<N_CHAN>(bridge_input, arr_packet);

        // 调用 galapagos_data_trans_chan
        galapagos_data_trans_chan<STREAM_TYPE, TRANS_IN_PACKET, N_CHAN, data_len>(arr_packet, arr_hls_data);

        // 调用 hls_data_write
        hls_data_write<STREAM_TYPE, TRANS_IN_PACKET, LOOP_SIZE, REMAIN_TRANS>(hls_output, arr_hls_data, i);
    }
}

template <class STREAM_TYPE, int elements_in_stream, int data_len>
void read_galapagos_interface_data(galapagos_interface *bridge_input)
{

    const int N_CHAN = STREAM_TYPE::size;   
    constexpr int TRANS_IN_PACKET = PACKET_DATA_LENGTH / data_len;  
    constexpr int LOOP_SIZE = nnet::ceil_div(elements_in_stream, TRANS_IN_PACKET);
    constexpr int REMAIN_TRANS = elements_in_stream - (LOOP_SIZE-1) * TRANS_IN_PACKET; 

    galapagos_packet arr_packet[N_CHAN];
	#pragma HLS ARRAY_PARTITION variable=arr_packet complete


    GALAPAGOS_2_HLS_LOOP:
    for (unsigned i = 0; i < LOOP_SIZE; i++) {
        // 调用 galapagos_data_read
        galapagos_data_read<N_CHAN>(bridge_input, arr_packet);
    }
}

template <class STREAM_TYPE, int elements_in_stream, unsigned int data_width, unsigned int chan>
void read_galapagos_interface_data(galapagos_interface *bridge_input)
{

    const int N_CHAN = chan;   
    constexpr int TRANS_IN_PACKET = PACKET_DATA_LENGTH / data_width;  
    constexpr int LOOP_SIZE = nnet::ceil_div(elements_in_stream, TRANS_IN_PACKET);
    constexpr int REMAIN_TRANS = elements_in_stream - (LOOP_SIZE-1) * TRANS_IN_PACKET; 

    galapagos_packet arr_packet[N_CHAN];
	#pragma HLS ARRAY_PARTITION variable=arr_packet complete


    GALAPAGOS_2_HLS_LOOP:
    for (unsigned i = 0; i < LOOP_SIZE; i++) {
        // 调用 galapagos_data_read
        galapagos_data_read<N_CHAN>(bridge_input, arr_packet);
    }
}






template <class STREAM_TYPE, int TRANS_IN_PACKET, int LOOP_SIZE, int REMAIN_TRANS>
void hls_data_read(hls::stream<STREAM_TYPE> &hls_input, STREAM_TYPE (&arr_hls_data)[TRANS_IN_PACKET], unsigned i) {
    #pragma HLS INLINE off
    HLS_DATA_READ:
    for (unsigned j = 0; j < TRANS_IN_PACKET; j++) {
        #pragma HLS PIPELINE
        arr_hls_data[j] = hls_input.read();

        if (i == LOOP_SIZE - 1 && j == REMAIN_TRANS - 1) {   // 边界条件，易错
            break;
        }
    }
}

template <class STREAM_TYPE, int TRANS_IN_PACKET, int N_CHAN, int data_len>
void hls_data_trans_chan(STREAM_TYPE (&arr_hls_data)[TRANS_IN_PACKET], galapagos_packet (&arr_packet)[N_CHAN]) {
    #pragma HLS INLINE off
    HLS_DATA_TRANS_CHAN:
    for (unsigned j = 0; j < N_CHAN; j++) {
        #pragma HLS PIPELINE
        HLS_DATA_TRANS_IN_PACKET:
        for (int k = 0; k < TRANS_IN_PACKET; k++) {
            #pragma HLS UNROLL
            arr_packet[j].data.range((k + 1) * data_len - 1, k * data_len) = arr_hls_data[k][j].range();
        }
    }
}

template <int N_CHAN, int PACKEN_MAX, int LOOP_SIZE>
void galapagos_data_write(galapagos_interface *bridge_output, galapagos_packet (&arr_packet)[N_CHAN], int &packet_count, unsigned i) {
    #pragma HLS INLINE off

	GALAPGOS_DATA_WRITE:
    for (unsigned j = 0; j < N_CHAN; j++) {
        #pragma HLS PIPELINE
        packet_count++; // 准备好一个packet
        arr_packet[j].last = (packet_count % PACKEN_MAX == 0) || (i == LOOP_SIZE - 1 && j == N_CHAN - 1);  // 边界条件，易错
        bridge_output->write(arr_packet[j]);
    }
}

template <class STREAM_TYPE, int elements_in_stream, int data_len>
void hls_stream_2_galapagos_interface(
    hls::stream<STREAM_TYPE> &hls_input,
    galapagos_interface *bridge_output,
    const int id,
    const int dest)
{
        #pragma HLS INLINE off

        const int N_CHAN = STREAM_TYPE::size;   //! 通道数，每一个hls_data[i]里包含的数的个数
        constexpr int TRANS_IN_PACKET = PACKET_DATA_LENGTH / data_len;  //! 一个packt可以存几个数，决定了并行读几次hls_data
        constexpr int LOOP_SIZE = nnet::ceil_div(elements_in_stream, TRANS_IN_PACKET);
        constexpr int REMAIN_TRANS = elements_in_stream - (LOOP_SIZE-1) * TRANS_IN_PACKET; //! REMAIN_TRANS一定小于TRANS_IN_PACKET
        const int PACKEN_MAX = 512;
        int packet_count = 0; // !追踪已发送的packet数,不知道要不要加static。
        //! 发送的包个数应为 LOOP_SIZE*特征图的通道数.    LOOP_SIZE = 流的长度 / TRANS_IN_PACKET 的上取整.   流的长度等于特征图的长*宽.

        STREAM_TYPE arr_hls_data[TRANS_IN_PACKET];
        #pragma HLS ARRAY_PARTITION variable=arr_hls_data complete
        
        galapagos_packet arr_packet[N_CHAN];
		#pragma HLS ARRAY_PARTITION variable=arr_packet complete
        
        for (unsigned i=0; i < N_CHAN; i++) {
            #pragma HLS UNROLL
            arr_packet[i].id = id;
            arr_packet[i].last = 0;
            arr_packet[i].keep = 0;//! 新增
            arr_packet[i].data = 0;
            arr_packet[i].dest = dest;
        }
        //! 新增，对arr_hls_data也进行初始化
        for (unsigned i=0; i < TRANS_IN_PACKET; i++) {
            #pragma HLS UNROLL 
            for (unsigned j=0; j < N_CHAN; j++) {
                #pragma HLS UNROLL
                arr_hls_data[i][j] = 0;
            }
        }

    HLS_2_GALAPAGOS_LOOP:
    for (unsigned i = 0; i < LOOP_SIZE; i++) {
        //#pragma HLS DATAFLOW   dataflow加了 RTL仿真不过,可能fifo大小不够。
        #pragma HLS PIPELINE
        // 调用 hls_data_read
        hls_data_read<STREAM_TYPE, TRANS_IN_PACKET, LOOP_SIZE, REMAIN_TRANS>(hls_input, arr_hls_data, i);
        // 调用 hls_data_trans_chan
        hls_data_trans_chan<STREAM_TYPE, TRANS_IN_PACKET, N_CHAN, data_len>(arr_hls_data, arr_packet);
        // 调用 galapagos_data_write
        galapagos_data_write<N_CHAN, PACKEN_MAX,LOOP_SIZE>(bridge_output, arr_packet, packet_count, i);
    }
}


// 新转换桥，galapagos packet -> stream group 
template <class STREAM_TYPE, int TRANS_IN_PACKET, int N_CHAN, int data_width>
void galapagos_data_trans_chan(const galapagos_packet (&arr_packet)[N_CHAN], STREAM_TYPE (&arr_hls_data)[TRANS_IN_PACKET][N_CHAN]) {
    #pragma HLS INLINE off

    GALA_DATA_TRANS_IN_PACKET:
    for (int k = 0; k < TRANS_IN_PACKET; k++) {
        #pragma HLS PIPELINE
        GALAPGOS_DATA_TRANS_CHAN:
        for (unsigned j = 0; j < N_CHAN; j++) {
            #pragma HLS UNROLL
            arr_hls_data[k][j].range() = arr_packet[j].data.range((k + 1) * data_width - 1, k * data_width);
        }
    }
}
template <class STREAM_TYPE, int TRANS_IN_PACKET, int LOOP_SIZE, int REMAIN_TRANS, int N_CHAN>
void hls_data_write(hls::stream<STREAM_TYPE> (&hls_output)[N_CHAN], STREAM_TYPE (&arr_hls_data)[TRANS_IN_PACKET][N_CHAN], unsigned i) {
    #pragma HLS INLINE off

    HLS_DATA_WRITE:
    for (unsigned j = 0; j < TRANS_IN_PACKET; j++) {
        #pragma HLS PIPELINE
        for (unsigned k = 0; k < N_CHAN; k++) {
            #pragma HLS UNROLL
            hls_output[k].write(arr_hls_data[j][k]);
        }
        if (i == LOOP_SIZE - 1 && j == REMAIN_TRANS - 1) {   // 边界条件，易错
            break;
        }
    }
}
template <class STREAM_TYPE, unsigned int elements_in_stream, unsigned int data_width, unsigned int chan>
void galapagos_interface_2_hls_stream(
    galapagos_interface *bridge_input,
    hls::stream<STREAM_TYPE> (&hls_output)[chan])
{

    #pragma HLS INLINE off

    const int N_CHAN = chan;   //! 通道数，每一个hls_data[i]里包含的数的个数
    constexpr int TRANS_IN_PACKET = PACKET_DATA_LENGTH / data_width;  //! 一个packt可以存几个数，决定了并行读几次hls_data
    constexpr int LOOP_SIZE = nnet::ceil_div(elements_in_stream, TRANS_IN_PACKET);
    constexpr int REMAIN_TRANS = elements_in_stream - (LOOP_SIZE-1) * TRANS_IN_PACKET; //! REMAIN_TRANS一定小于TRANS_IN_PACKET

    galapagos_packet arr_packet[N_CHAN];
	#pragma HLS ARRAY_PARTITION variable=arr_packet complete

    STREAM_TYPE arr_hls_data[TRANS_IN_PACKET][N_CHAN];
	#pragma HLS ARRAY_PARTITION variable=arr_hls_data type=cyclic factor=N_CHAN dim=2


    GALAPAGOS_2_HLS_LOOP:
    for (unsigned i = 0; i < LOOP_SIZE; i++) {
        //#pragma HLS DATAFLOW
        #pragma HLS PIPELINE

        // 调用 galapagos_data_read
        galapagos_data_read<N_CHAN>(bridge_input, arr_packet);

        // 调用 galapagos_data_trans_chan
        galapagos_data_trans_chan<STREAM_TYPE, TRANS_IN_PACKET, N_CHAN, data_width>(arr_packet, arr_hls_data);

        // 调用 hls_data_write
        hls_data_write<STREAM_TYPE, TRANS_IN_PACKET, LOOP_SIZE, REMAIN_TRANS, N_CHAN>(hls_output, arr_hls_data, i);
    }
}


// 新转换桥，stream group -> galapagos packet 
template <class STREAM_TYPE, int TRANS_IN_PACKET, int LOOP_SIZE, int REMAIN_TRANS, int N_CHAN>
void hls_data_read(hls::stream<STREAM_TYPE> (&hls_input)[N_CHAN], STREAM_TYPE (&arr_hls_data)[TRANS_IN_PACKET][N_CHAN], unsigned i) {
    #pragma HLS INLINE off
    HLS_DATA_READ:
    for (unsigned j = 0; j < TRANS_IN_PACKET; j++) {
        #pragma HLS PIPELINE
        for (unsigned k = 0; k < N_CHAN; k++) {
            #pragma HLS UNROLL
            arr_hls_data[j][k] = hls_input[k].read();
        }

        if (i == LOOP_SIZE - 1 && j == REMAIN_TRANS - 1) {   // 边界条件，易错
            break;
        }
    }
}
template <class STREAM_TYPE, int TRANS_IN_PACKET, int N_CHAN, int data_width>
void hls_data_trans_chan(STREAM_TYPE (&arr_hls_data)[TRANS_IN_PACKET][N_CHAN], galapagos_packet (&arr_packet)[N_CHAN]) {
    #pragma HLS INLINE off

    HLS_DATA_TRANS_IN_PACKET:
    for (int k = 0; k < TRANS_IN_PACKET; k++) {
    #pragma HLS PIPELINE
        HLS_DATA_TRANS_CHAN:
        for (unsigned j = 0; j < N_CHAN; j++) {
            #pragma HLS UNROLL
            arr_packet[j].data.range((k + 1) * data_width - 1, k * data_width) = arr_hls_data[k][j].range();
        }
    }
}

template <class STREAM_TYPE, int elements_in_stream, int data_width, int chan>
void hls_stream_2_galapagos_interface(
    hls::stream<STREAM_TYPE> (&hls_input)[chan],
    galapagos_interface *bridge_output,
    const int id,
    const int dest)
{
        #pragma HLS INLINE off

        const int N_CHAN = chan;   //! 通道数，每一个hls_data[i]里包含的数的个数
        constexpr int TRANS_IN_PACKET = PACKET_DATA_LENGTH / data_width;  //! 一个packt可以存几个数，决定了并行读几次hls_data
        constexpr int LOOP_SIZE = nnet::ceil_div(elements_in_stream, TRANS_IN_PACKET);
        constexpr int REMAIN_TRANS = elements_in_stream - (LOOP_SIZE-1) * TRANS_IN_PACKET; //! REMAIN_TRANS一定小于TRANS_IN_PACKET
        const int PACKEN_MAX = 512;
        int packet_count = 0; // !追踪已发送的packet数,不知道要不要加static。
        //! 发送的包个数应为 LOOP_SIZE*特征图的通道数.    LOOP_SIZE = 流的长度 / TRANS_IN_PACKET 的上取整.   流的长度等于特征图的长*宽.

        STREAM_TYPE arr_hls_data[TRANS_IN_PACKET][N_CHAN];
        #pragma HLS ARRAY_RESHAPE variable=arr_hls_data type=cyclic factor=N_CHAN dim=2
        // #pragma HLS ARRAY_PARTITION variable=arr_hls_data complete
        
        galapagos_packet arr_packet[N_CHAN];
		#pragma HLS ARRAY_PARTITION variable=arr_packet complete
        
        for (unsigned i=0; i < N_CHAN; i++) {
            #pragma HLS UNROLL
            arr_packet[i].id = id;
            arr_packet[i].last = 0;
            arr_packet[i].keep = 0;//! 新增
            arr_packet[i].data = 0;
            arr_packet[i].dest = dest;
        }
        //! 新增，对arr_hls_data也进行初始化
        for (unsigned i=0; i < TRANS_IN_PACKET; i++) {
            #pragma HLS UNROLL 
            for (unsigned j=0; j < N_CHAN; j++) {
                #pragma HLS UNROLL
                arr_hls_data[i][j] = 0;
            }
        }

    HLS_2_GALAPAGOS_LOOP:
    for (unsigned i = 0; i < LOOP_SIZE; i++) {
        //#pragma HLS DATAFLOW   dataflow加了 RTL仿真不过,可能fifo大小不够。
        #pragma HLS PIPELINE
        // 调用 hls_data_read
        hls_data_read<STREAM_TYPE, TRANS_IN_PACKET, LOOP_SIZE, REMAIN_TRANS, N_CHAN>(hls_input, arr_hls_data, i);
        // 调用 hls_data_trans_chan
        hls_data_trans_chan<STREAM_TYPE, TRANS_IN_PACKET, N_CHAN, data_width>(arr_hls_data, arr_packet);
        // 调用 galapagos_data_write
        galapagos_data_write<N_CHAN, PACKEN_MAX,LOOP_SIZE>(bridge_output, arr_packet, packet_count, i);
    }
}

// 转换桥，galapagos packet -> stream<array> group 

template <class STREAM_TYPE, int TRANS_IN_PACKET, int LOOP_SIZE, int REMAIN_TRANS, int N_CHAN>
void hls_data_write(hls::stream<STREAM_TYPE> (&hls_output)[N_CHAN/STREAM_TYPE::size], typename STREAM_TYPE::value_type (&arr_hls_data)[TRANS_IN_PACKET][N_CHAN], unsigned i) {
    #pragma HLS INLINE off

    STREAM_TYPE hls_data[N_CHAN/STREAM_TYPE::size];
    HLS_DATA_WRITE:
    for (unsigned j = 0; j < TRANS_IN_PACKET; j++) {
        #pragma HLS PIPELINE
        for (unsigned k = 0; k < N_CHAN/STREAM_TYPE::size; k++) {
            #pragma HLS UNROLL
            for (unsigned l = 0; l < STREAM_TYPE::size; l++) {
                #pragma HLS UNROLL
                hls_data[k][l] = arr_hls_data[j][k*STREAM_TYPE::size + l];
            }
            hls_output[k].write(hls_data[k]);
        }
        if (i == LOOP_SIZE - 1 && j == REMAIN_TRANS - 1) {   // 边界条件，易错
            break;
        }
    }
}

template <class STREAM_TYPE, unsigned int elements_in_stream, unsigned int data_width, unsigned int chan>
void galapagos_interface_2_hls_stream(
    galapagos_interface *bridge_input,
    hls::stream<STREAM_TYPE> (&hls_output)[chan/STREAM_TYPE::size])
{

    #pragma HLS INLINE off

    const int N_CHAN = chan;   //! 通道数，每一个hls_data[i]里包含的数的个数
    constexpr int TRANS_IN_PACKET = PACKET_DATA_LENGTH / data_width;  //! 一个packt可以存几个数，决定了并行读几次hls_data
    constexpr int LOOP_SIZE = nnet::ceil_div(elements_in_stream, TRANS_IN_PACKET);
    constexpr int REMAIN_TRANS = elements_in_stream - (LOOP_SIZE-1) * TRANS_IN_PACKET; //! REMAIN_TRANS一定小于TRANS_IN_PACKET

    galapagos_packet arr_packet[N_CHAN];
	#pragma HLS ARRAY_PARTITION variable=arr_packet complete

    typename STREAM_TYPE::value_type arr_hls_data[TRANS_IN_PACKET][N_CHAN];
	#pragma HLS ARRAY_RESHAPE variable=arr_hls_data type=cyclic factor=N_CHAN dim=2


    GALAPAGOS_2_HLS_LOOP:
    for (unsigned i = 0; i < LOOP_SIZE; i++) {
        //#pragma HLS DATAFLOW
        #pragma HLS PIPELINE

        // 调用 galapagos_data_read
        galapagos_data_read<N_CHAN>(bridge_input, arr_packet);

        // 调用 galapagos_data_trans_chan
        galapagos_data_trans_chan<typename STREAM_TYPE::value_type, TRANS_IN_PACKET, N_CHAN, data_width>(arr_packet, arr_hls_data);

        // 调用 hls_data_write
        hls_data_write<STREAM_TYPE, TRANS_IN_PACKET, LOOP_SIZE, REMAIN_TRANS, N_CHAN>(hls_output, arr_hls_data, i);
    }
}

// 转换桥，stream<array> group -> galapagos packet

template <class STREAM_TYPE, int TRANS_IN_PACKET, int LOOP_SIZE, int REMAIN_TRANS, int N_CHAN>
void hls_data_read(hls::stream<STREAM_TYPE> (&hls_input)[N_CHAN/STREAM_TYPE::size], typename STREAM_TYPE::value_type (&arr_hls_data)[TRANS_IN_PACKET][N_CHAN], unsigned i) {
    #pragma HLS INLINE off

    STREAM_TYPE hls_data[N_CHAN/STREAM_TYPE::size];
    HLS_DATA_READ:
    for (unsigned j = 0; j < TRANS_IN_PACKET; j++) {
        #pragma HLS PIPELINE
        for (unsigned k = 0; k < N_CHAN/STREAM_TYPE::size; k++) {
            #pragma HLS UNROLL
            hls_data[k] = hls_input[k].read();
            for (unsigned l = 0; l < STREAM_TYPE::size; l++) {
                #pragma HLS UNROLL
                arr_hls_data[j][k*STREAM_TYPE::size+l] = hls_data[k][l];
            }
        }

        if (i == LOOP_SIZE - 1 && j == REMAIN_TRANS - 1) {   // 边界条件，易错
            break;
        }
    }
}

template <class STREAM_TYPE, int elements_in_stream, int data_width, int chan>
void hls_stream_2_galapagos_interface(
    hls::stream<STREAM_TYPE> (&hls_input)[chan/STREAM_TYPE::size],
    galapagos_interface *bridge_output,
    const int id,
    const int dest)
{
        #pragma HLS INLINE off

        const int N_CHAN = chan;   //! 通道数，每一个hls_data[i]里包含的数的个数
        constexpr int TRANS_IN_PACKET = PACKET_DATA_LENGTH / data_width;  //! 一个packt可以存几个数，决定了并行读几次hls_data
        constexpr int LOOP_SIZE = nnet::ceil_div(elements_in_stream, TRANS_IN_PACKET);
        constexpr int REMAIN_TRANS = elements_in_stream - (LOOP_SIZE-1) * TRANS_IN_PACKET; //! REMAIN_TRANS一定小于TRANS_IN_PACKET
        const int PACKEN_MAX = 512;
        int packet_count = 0; // !追踪已发送的packet数,不知道要不要加static。
        //! 发送的包个数应为 LOOP_SIZE*特征图的通道数.    LOOP_SIZE = 流的长度 / TRANS_IN_PACKET 的上取整.   流的长度等于特征图的长*宽.

        typename STREAM_TYPE::value_type arr_hls_data[TRANS_IN_PACKET][N_CHAN];
        #pragma HLS ARRAY_RESHAPE variable=arr_hls_data type=cyclic factor=N_CHAN dim=2
        // #pragma HLS ARRAY_PARTITION variable=arr_hls_data complete
        
        galapagos_packet arr_packet[N_CHAN];
		#pragma HLS ARRAY_PARTITION variable=arr_packet complete
        
        for (unsigned i=0; i < N_CHAN; i++) {
            #pragma HLS UNROLL
            arr_packet[i].id = id;
            arr_packet[i].last = 0;
            arr_packet[i].keep = 0;//! 新增
            arr_packet[i].data = 0;
            arr_packet[i].dest = dest;
        }
        //! 新增，对arr_hls_data也进行初始化
        for (unsigned i=0; i < TRANS_IN_PACKET; i++) {
            #pragma HLS UNROLL 
            for (unsigned j=0; j < N_CHAN; j++) {
                #pragma HLS UNROLL
                arr_hls_data[i][j] = 0;
            }
        }

    HLS_2_GALAPAGOS_LOOP:
    for (unsigned i = 0; i < LOOP_SIZE; i++) {
        //#pragma HLS DATAFLOW   dataflow加了 RTL仿真不过,可能fifo大小不够。
        #pragma HLS PIPELINE
        // 调用 hls_data_read
        hls_data_read<STREAM_TYPE, TRANS_IN_PACKET, LOOP_SIZE, REMAIN_TRANS, N_CHAN>(hls_input, arr_hls_data, i);
        // 调用 hls_data_trans_chan
        hls_data_trans_chan<typename STREAM_TYPE::value_type, TRANS_IN_PACKET, N_CHAN, data_width>(arr_hls_data, arr_packet);
        // 调用 galapagos_data_write
        galapagos_data_write<N_CHAN, PACKEN_MAX,LOOP_SIZE>(bridge_output, arr_packet, packet_count, i);
    }
}

}
#endif