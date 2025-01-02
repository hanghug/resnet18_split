#ifndef NNET_HELPERS_H
#define NNET_HELPERS_H

#include "hls_stream.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

namespace nnet {

#ifndef __SYNTHESIS__

#ifndef WEIGHTS_DIR
#define WEIGHTS_DIR "firmware/weights"
#endif

template <class T, size_t SIZE> void load_weights_from_txt(T *w, const char *fname) {

    std::string full_path = std::string(WEIGHTS_DIR) + "/" + std::string(fname);
    std::ifstream infile(full_path.c_str(), std::ios::binary);

    if (infile.fail()) {
        std::cerr << "ERROR: file " << std::string(fname) << " does not exist" << std::endl;
        exit(1);
    }

    std::string line;
    if (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string token;

        size_t i = 0;
        while (std::getline(iss, token, ',')) {
            std::istringstream(token) >> w[i];
            i++;
        }

        if (SIZE != i) {
            std::cerr << "ERROR: Expected " << SIZE << " values";
            std::cerr << " but read only " << i << " values" << std::endl;
        }
    }
}

template <class T, size_t SIZE> void load_compressed_weights_from_txt(T *w, const char *fname) {

    std::string full_path = std::string(WEIGHTS_DIR) + "/" + std::string(fname);
    std::ifstream infile(full_path.c_str(), std::ios::binary);

    if (infile.fail()) {
        std::cerr << "ERROR: file " << std::string(fname) << " does not exist" << std::endl;
        exit(1);
    }

    std::string line;
    if (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string token;
        std::string extra_chars = "} ";

        size_t i = 0;
        while (std::getline(iss, token, '{')) {
            if (token.length() == 0) {
                continue;
            }
            for (char c : extra_chars) {
                token.erase(std::remove(token.begin(), token.end(), c), token.end());
            }
            if (token.back() == ',') {
                token.erase(token.end() - 1);
            }

            std::replace(token.begin(), token.end(), ',', ' ');
            std::istringstream structss(token);

            if (!(structss >> w[i].row_index >> w[i].col_index >> w[i].weight)) {
                std::cerr << "ERROR: Unable to parse file " << std::string(fname);
                exit(1);
            }
            i++;
        }

        if (SIZE != i) {
            std::cerr << "ERROR: Expected " << SIZE << " values";
            std::cerr << " but read only " << i << " values" << std::endl;
        }
    }
}

template <class T, size_t SIZE> void load_exponent_weights_from_txt(T *w, const char *fname) {

    std::string full_path = std::string(WEIGHTS_DIR) + "/" + std::string(fname);
    std::ifstream infile(full_path.c_str(), std::ios::binary);

    if (infile.fail()) {
        std::cerr << "ERROR: file " << std::string(fname) << " does not exist" << std::endl;
        exit(1);
    }

    std::string line;
    if (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string token;
        std::string extra_chars = "} ";

        size_t i = 0;
        while (std::getline(iss, token, '{')) {
            if (token.length() == 0) {
                continue;
            }
            for (char c : extra_chars) {
                token.erase(std::remove(token.begin(), token.end(), c), token.end());
            }
            if (token.back() == ',') {
                token.erase(token.end() - 1);
            }

            std::replace(token.begin(), token.end(), ',', ' ');
            std::istringstream structss(token);

            if (!(structss >> w[i].sign >> w[i].weight)) {
                std::cerr << "ERROR: Unable to parse file " << std::string(fname);
                exit(1);
            }
            i++;
        }

        if (SIZE != i) {
            std::cerr << "ERROR: Expected " << SIZE << " values";
            std::cerr << " but read only " << i << " values" << std::endl;
        }
    }
}
template <class srcType, class dstType, size_t SIZE> void convert_data(srcType *src, dstType *dst) {
    for (size_t i = 0; i < SIZE; i++) {
        dst[i] = dstType(src[i]);
    }
}

template <class srcType, class dstType, size_t SIZE> 
void convert_data(srcType *src, hls::stream<dstType> &dst) {
    for (size_t i = 0; i < SIZE / dstType::size; i++) {
        dstType ctype;
        for (size_t j = 0; j < dstType::size; j++) {
            ctype[j] = typename dstType::value_type(src[i * dstType::size + j]);
        }
        dst.write(ctype);
    }
}

//! 数组到stream流并行数组
template <class srcType, class dstType, size_t SIZE, unsigned int chan> 
void convert_data(srcType *src, hls::stream<dstType> (&dst)[chan]) {
    for (size_t i = 0; i < SIZE / chan; i++) {
        for (size_t j = 0; j < chan; j++) {
            dstType temp;
            temp = static_cast<dstType>(src[i * chan + j]);
            dst[j].write(temp);
        }
    }
}



template <class srcType, class dstType, size_t SIZE> 
void convert_data(hls::stream<srcType> &src, dstType *dst) {
    for (size_t i = 0; i < SIZE / srcType::size; i++) {
        srcType ctype = src.read();
        for (size_t j = 0; j < srcType::size; j++) {
            dst[i * srcType::size + j] = dstType(ctype[j]);
        }
    }
}

//! stream流并行数组到数组
template <class srcType, class dstType, size_t SIZE, unsigned int chan> 
void convert_data(hls::stream<srcType> (&src)[chan], dstType *dst) {
    for (size_t i = 0; i < SIZE / chan; i++) {
        srcType ctype[chan];
        for (size_t j = 0; j < chan; j++) {
            ctype[j] = src[j].read();
        }
        for (size_t j = 0; j < chan; j++) {
            dst[i * chan + j] = dstType(ctype[j]);
        }
    }
}



//! add
template <class srcType, class dstType, size_t SIZE> 
void convert_data(hls::stream<srcType> &src, dstType *dst, bool keep) {
    for (size_t i = 0; i < SIZE / srcType::size; i++) {
        srcType ctype = src.read();
        for (size_t j = 0; j < srcType::size; j++) {
            dst[i * srcType::size + j] = dstType(ctype[j]);
        }
        if (keep)
            src.write(ctype);
    }
}



extern bool trace_enabled;
extern std::map<std::string, void *> *trace_outputs;
extern size_t trace_type_size;

template <class data_T, class save_T> void save_output_array(data_T *data, save_T *ptr, size_t layer_size) {
    for (int i = 0; i < layer_size; i++) {
        ptr[i] = save_T(data[i]);
    }
}

template <class data_T, class save_T> void save_output_array(hls::stream<data_T> &data, save_T *ptr, size_t layer_size) {
    for (size_t i = 0; i < layer_size / data_T::size; i++) {
        data_T ctype = data.read();
        for (size_t j = 0; j < data_T::size; j++) {
            ptr[i * data_T::size + j] = save_T(ctype[j]);
        }
        data.write(ctype);
    }
}

// We don't want to include save_T in this function because it will be inserted into myproject.cpp
// so a workaround with element size is used
template <class data_T> void save_layer_output(data_T *data, const char *layer_name, size_t layer_size) {
    if (!trace_enabled)
        return;

    if (trace_outputs) {
        if (trace_outputs->count(layer_name) > 0) {
            if (trace_type_size == 4) {
                save_output_array<data_T, float>(data, (float *)(*trace_outputs)[layer_name], layer_size);
            } else if (trace_type_size == 8) {
                save_output_array<data_T, double>(data, (double *)(*trace_outputs)[layer_name], layer_size);
            } else {
                std::cout << "Unknown trace type!" << std::endl;
            }
        } else {
            std::cout << "Layer name: " << layer_name << " not found in debug storage!" << std::endl;
        }
    } else {
        std::ostringstream filename;
        filename << "./tb_data/" << layer_name << "_output.log"; // TODO if run as a shared lib, path should be ../tb_data
        std::fstream out;
        out.open(filename.str(), std::ios::app);
        assert(out.is_open());
        for (int i = 0; i < layer_size; i++) {
            out << float(data[i]) << " "; // We don't care about precision in text files
        }
        out << std::endl;
        out.close();
    }
}

template <class data_T> 
void save_layer_output(hls::stream<data_T> &data, const char *layer_name, size_t layer_size) {
    if (!trace_enabled)
        return;

    if (trace_outputs) {
        if (trace_outputs->count(layer_name) > 0) {
            if (trace_type_size == 4) {
                save_output_array<data_T, float>(data, (float *)(*trace_outputs)[layer_name], layer_size);
            } else if (trace_type_size == 8) {
                save_output_array<data_T, double>(data, (double *)(*trace_outputs)[layer_name], layer_size);
            } else {
                std::cout << "Unknown trace type!" << std::endl;
            }
        } else {
            std::cout << "Layer name: " << layer_name << " not found in debug storage!" << std::endl;
        }
    } else {
        std::ostringstream filename;
        filename << "./tb_data/" << layer_name << "_output.log"; // TODO if run as a shared lib, path should be ../tb_data
        std::fstream out;
        out.open(filename.str(), std::ios::app);
        assert(out.is_open());
        for (size_t i = 0; i < layer_size / data_T::size; i++) {
            data_T ctype = data.read();
            for (size_t j = 0; j < data_T::size; j++) {
                out << float(ctype[j]) << " "; // We don't care about precision in text files
            }
            data.write(ctype);
        }
        out << std::endl;
        out.close();
    }
}

//! Save layer output to stream并行数组
template <class data_T, class save_T, unsigned int chan> 
void save_output_array(hls::stream<data_T> (&data)[chan], save_T *ptr, size_t layer_size) {
    for (size_t i = 0; i < layer_size / chan; i++) {
        data_T ctype[chan];
        for (size_t j = 0; j < chan; j++) {
            ctype[j] = data[j].read(); 
        }

        for (size_t j = 0; j < chan; j++) {
            ptr[i * chan + j] = save_T(ctype[j]);
        }

        for (int j = 0; j < chan; j++) {  //! 写入
            data[j].write(ctype[j]);    
        }
    }
}


template <class data_T, unsigned int chan> 
void save_layer_output(hls::stream<data_T> (&data)[chan], const char *layer_name, size_t layer_size) {
    if (!trace_enabled)
        return;

    if (trace_outputs) {
        if (trace_outputs->count(layer_name) > 0) {
            if (trace_type_size == 4) {
                save_output_array<data_T, float, chan>(data, (float *)(*trace_outputs)[layer_name], layer_size);
            } else if (trace_type_size == 8) {
                save_output_array<data_T, double, chan>(data, (double *)(*trace_outputs)[layer_name], layer_size);
            } else {
                std::cout << "Unknown trace type!" << std::endl;
            }
        } else {
            std::cout << "Layer name: " << layer_name << " not found in debug storage!" << std::endl;
        }
    } else {
        std::ostringstream filename;
        filename << "./tb_data/" << layer_name << "_output.log"; // TODO if run as a shared lib, path should be ../tb_data
        std::fstream out;
        out.open(filename.str(), std::ios::app);
        assert(out.is_open());
        for (size_t i = 0; i < layer_size / chan; i++) {

            data_T ctype[chan];
            for (int j = 0; j < chan; j++) {  //! 读出
                ctype[j] = data[j].read();    
            }

            for (size_t j = 0; j < chan; j++) {
                out << float(ctype[j]) << " "; // We don't care about precision in text files
            }

            for (int j = 0; j < chan; j++) {  //! 写入
                data[j].write(ctype[j]);    
            }
        }
        out << std::endl;
        out.close();
    }
}


#endif   //!#ifndef __SYNTHESIS__

template <class src_T, class dst_T, size_t OFFSET, size_t SIZE> 
void copy_data(std::vector<src_T> src, dst_T dst[SIZE]) {
    typename std::vector<src_T>::const_iterator in_begin = src.cbegin() + OFFSET;
    typename std::vector<src_T>::const_iterator in_end = in_begin + SIZE;
    std::copy(in_begin, in_end, dst);
}

//! vector数组拷贝到hls_stream流中，支持float/double到ap_fixed<>的拷贝
template <class src_T, class dst_T, size_t OFFSET, size_t SIZE>
void copy_data(std::vector<src_T> src, hls::stream<dst_T> &dst) {
    typename std::vector<src_T>::const_iterator in_begin = src.cbegin() + OFFSET;
    typename std::vector<src_T>::const_iterator in_end = in_begin + SIZE;

    size_t i_pack = 0;
    dst_T dst_pack;
    for (typename std::vector<src_T>::const_iterator i = in_begin; i != in_end; ++i) {
        dst_pack[i_pack++] = typename dst_T::value_type(*i);//!如果 src_T 和 dst_T 类型之间有不同的位宽，需要确保数据在转换时不会丢失精度。因为这是赋值操作，不是按位复制。
        if (i_pack == dst_T::size) {
            i_pack = 0;
            dst.write(dst_pack);
        }
    }
}
//! vector数组拷贝到hls_stream流组
template <class src_T, class dst_T, size_t OFFSET, size_t SIZE, unsigned int chan>
void copy_data(std::vector<src_T> src, hls::stream<dst_T> (&dst)[chan]) {
    typename std::vector<src_T>::const_iterator in_begin = src.cbegin() + OFFSET;
    typename std::vector<src_T>::const_iterator in_end = in_begin + SIZE;
    size_t i_pack = 0;
    for (typename std::vector<src_T>::const_iterator i = in_begin; i != in_end; ++i) {
        dst[i_pack++].write(dst_T(*i));
        if (i_pack == chan) {
            i_pack = 0;
        }
    }
}

//! vector数组拷贝到hls_stream<array>流组
template <class src_T, class dst_T, size_t OFFSET, size_t SIZE, unsigned int chan>
void copy_data(std::vector<src_T> src, hls::stream<dst_T> (&dst)[chan/dst_T::size]) {
    typename std::vector<src_T>::const_iterator in_begin = src.cbegin() + OFFSET;
    typename std::vector<src_T>::const_iterator in_end = in_begin + SIZE;
    size_t i_pack = 0;
    dst_T dst_data;

    for (uint j = 0; j < SIZE/chan; j++){
        for (uint g = 0; g < chan/dst_T::size; g++){
            for (uint s = 0; s < dst_T::size; s++){
                dst_data[s] = static_cast<typename dst_T::value_type>(*(in_begin + j*chan + g*dst_T::size + s));
            }
            dst[g].write(dst_data);
        }
    }
}

template <class src_T, class dst_T, size_t OFFSET, size_t SIZE> void copy_data_axi(std::vector<src_T> src, dst_T dst[SIZE]) {
    for (auto i = 0; i < SIZE; i++)
        if (i == SIZE - 1) {
            dst[i].data = src[i];
            dst[i].last = 1;
        } else {
            dst[i].data = src[i];
            dst[i].last = 0;
        }
}

template <class res_T, size_t SIZE> void print_result(res_T result[SIZE], std::ostream &out, bool keep = false) {
    for (int i = 0; i < SIZE; i++) {            //!为了保持重载，keep这里用不到。
        out << result[i] << " ";
    }
    out << std::endl;
}

template <class res_T, size_t SIZE> void print_result(hls::stream<res_T> &result, std::ostream &out, bool keep = false) {
    for (int i = 0; i < SIZE / res_T::size; i++) {
        res_T res_pack = result.read(); //!从流中读出以后，数据就不在了，因此为设置Keep为true，则重新写入。
        for (int j = 0; j < res_T::size; j++) {
            out << res_pack[j] << " ";
        }
        if (keep)
            result.write(res_pack);
    }
    out << std::endl;
}

template <class res_T, size_t SIZE, unsigned int chan> void print_result(hls::stream<res_T> (&result)[chan], std::ostream &out, bool keep = false) {
    for (int i = 0; i < SIZE / chan; i++) {
        for (int j = 0; j < chan; j++) {
            res_T res = result[j].read();
            out << res << " ";
            if (keep) result[j].write(res);
        }
        
    }
    out << std::endl;
}

template <class res_T, size_t SIZE, unsigned int chan> void print_result(hls::stream<res_T> (&result)[chan/res_T::size], std::ostream &out, bool keep = false) {
    for (int i = 0; i < SIZE / chan; i++) {
        for (int j = 0; j < chan/res_T::size; j++) {
            res_T res = result[j].read();
            for (int k = 0; k < res_T::size; k++) {
                out << res[k] << " ";
            }
            if (keep) result[j].write(res);
        }
    }
    out << std::endl;
}

template <class data_T, size_t SIZE> void fill_zero(data_T data[SIZE]) { std::fill_n(data, SIZE, 0.); }

template <class data_T, size_t SIZE> void fill_zero(hls::stream<data_T> &data) {
    for (int i = 0; i < SIZE / data_T::size; i++) {
        data_T data_pack;
        for (int j = 0; j < data_T::size; j++) {
            data_pack[j] = 0.;
        }
        data.write(data_pack);
    }
}

// 流组
template <class data_T, size_t SIZE, unsigned int chan> void fill_zero(hls::stream<data_T> (&data)[chan]) {
    for (int i = 0; i < SIZE / chan; i++) {
        for (int j = 0; j < chan; j++) {
            data[j].write(0.);
        }
    }
}

template <class data_T, size_t SIZE, unsigned int chan> void fill_zero(hls::stream<data_T> (&data)[chan/data_T::size]) {
    for (int i = 0; i < SIZE / chan; i++) {
        for (int j = 0; j < chan/data_T::size; j++) {
            data_T data_pack;
            for (int k = 0; k < data_T::size; k++){
                data_pack[k] = 0.;
            }
            data[j].write(data_pack);
        }
    }
}











//!add 初始化 从0 1 2 3...递增赋值
template<class data_T, size_t OFFSET, size_t SIZE>
void fill_init(hls::stream<data_T> &data) {
    for(int i = 0; i < SIZE / data_T::size; i++) { //SIZE / data_T::size行
        data_T data_pack;
        for(int j = 0; j < data_T::size; j++) { //data_T::size列
            data_pack[j] = (OFFSET+j+i*data_T::size)*1.0;
        }
        data.write(data_pack);
    }
}


/*
在循环中使用 push_back 可能会导致性能下降，因为每次添加元素都需要重新分配内存。
如果你知道 dst 的大小，最好在使用前使用 dst.reserve(SIZE) 来预分配足够的内存，然后使用索引直接赋值，而不是使用 push_back。
std::vector::reserve 函数用于预分配 std::vector 的内存空间，以便存储至少指定数量的元素，
但它并不会改变 std::vector 中已经存在的元素数量。换句话说，调用 reserve 不会影响已有的数据。
如果想要清空 std::vector 中的数据，可以使用 std::vector::clear 函数
*/

//! 从一维数组到vector数组的赋值。函数重载  用于每次追加vector，不需要注明追加的起始地址start，直接push,效率偏低。
template <class src_T, class dst_T, size_t OFFSET, size_t SIZE>
void copy_data(src_T src[SIZE], std::vector<dst_T> &dst) {
    for (size_t i = 0; i < SIZE; ++i) {
        dst.push_back(static_cast<dst_T>(src[i + OFFSET])); 
    }
}

//! 从一维数组到vector数组的赋值。函数重载  用于每次追加vector，需要注明追加的起始地址start。也支持覆盖。
template <class src_T, class dst_T, size_t OFFSET, size_t SIZE>  //! 多定义了一个start参数，不能参数重载了。效率高。
void copy_data(src_T src[SIZE], std::vector<dst_T> &dst, size_t start) {
    size_t required_size = start + SIZE;
    
    if (dst.size() < required_size) {
        dst.resize(required_size);
    }

    for (size_t i = 0; i < SIZE; ++i) {
        dst[start + i] = static_cast<dst_T>(src[i + OFFSET]);
    }
}

//! 一维数组到stream流转换。函数重载。 offset写0，size写数组长度，进行全部转换
template <class src_T, class dst_T, size_t OFFSET, size_t SIZE>
void copy_data(src_T src[SIZE], hls::stream<dst_T> &dst_stream) {
    dst_T dst_data;
    size_t i_pack = 0;

    #pragma HLS inline

    for (size_t i = 0; i < SIZE; ++i) {
        dst_data[i_pack++] = static_cast<typename dst_T::value_type>(src[i + OFFSET]);

        if (i_pack == dst_T::size) {
            i_pack = 0;
            dst_stream.write(dst_data);
        }
    }
}

//! 一维数组到stream流组转换。函数重载。 offset写0，size写数组长度，进行全部转换
template <class src_T, class dst_T, size_t OFFSET, size_t SIZE, unsigned int chan>
void copy_data(src_T src[SIZE], hls::stream<dst_T> (&dst_stream)[chan]) {
    size_t i_pack = 0;

    #pragma HLS inline

    for (size_t i = 0; i < SIZE; ++i) {
        dst_stream[i_pack++].write(static_cast<dst_T>(src[i + OFFSET]));

        if (i_pack == chan) {
            i_pack = 0;
        }
    }
}

//! 一维数组到array stream group转换。函数重载。 offset写0，size写数组长度，进行全部转换
template <class src_T, class dst_T, size_t OFFSET, size_t SIZE, unsigned int chan>
void copy_data(src_T src[SIZE], hls::stream<dst_T> (&dst_stream)[chan/dst_T::size]) {
    dst_T dst_data;

    #pragma HLS inline

    for (uint j = 0; j < SIZE/chan; j++){
        for (uint g = 0; g < chan/dst_T::size; g++){
            for (uint s = 0; s < dst_T::size; s++){
                dst_data[s] = static_cast<typename dst_T::value_type>(src[OFFSET + j*chan + g*dst_T::size + s]);
            }
            dst_stream[g].write(dst_data);
        }
    }

}


//! 输出vector数组 重载函数。 更新后这里只需要一个参数了，因为它的大小可以自动获得。
template<class res_T>
void print_result(std::vector<res_T> &result, std::ostream &out) {
    for(const auto &item : result) {
        out << item << " ";
    }
    out << std::endl;
}


//! 使用此函数务必确保part_len的长度要大于input_stream的长度
template<class T, int part_len>
int extract_stream(hls::stream<T> &output_stream, hls::stream<T> &input_stream) {
    for (int i = 0; i < part_len; ++i) {
        #pragma HLS pipeline II=1
        if (!input_stream.empty()) {
            T data = input_stream.read();
            output_stream.write(data);
        }   else {
            ;
        }
    }
    return output_stream.size();  //!方便判断是否成功读取了part_len个。
} //! 还有一定需要注意，该代码在执行时需要确定原来的input_stream是否为空，如果不为空，它也是直接写入了，就算成功读出来也不是part_len的数值。









// template <class STREAM_T, typename CONFIG_T> 
// void split2(hls::stream<STREAM_T> (&input_stream)[CONFIG_T::n_chan*2], hls::stream<STREAM_T> (&output_stream1)[CONFIG_T::n_chan], hls::stream<STREAM_T> (&output_stream2)[CONFIG_T::n_chan]) {
//     for (int i = 0; i < CONFIG_T::in_height*CONFIG_T::in_width; i++) {
//         #pragma HLS PIPELINE
//         for (int j = 0; j < CONFIG_T::n_chan; j++) {
//             #pragma HLS UNROLL
//             output_stream1[j].write(input_stream[j].read());
//             output_stream2[j].write(input_stream[j+CONFIG_T::n_chan].read());
//         }
//     }
// }

// template <class STREAM_T, typename CONFIG_T> 
// void merge2(hls::stream<STREAM_T> (&input_stream1)[CONFIG_T::n_filt], hls::stream<STREAM_T> (&input_stream2)[CONFIG_T::n_filt], hls::stream<STREAM_T> (&output_stream)[CONFIG_T::n_filt]) {
//     for (int i = 0; i < CONFIG_T::out_height*CONFIG_T::out_width; i++) {
//         #pragma HLS PIPELINE
//         for (int j = 0; j < CONFIG_T::n_filt; j++) {
//             #pragma HLS UNROLL
//             output_stream[j].write(input_stream1[j].read()+input_stream2[j].read());
//         }
//     }
// }

// split to 2 output, from nnet_merge_stream.h::split3
template <class src_T, class res_T, typename CONFIG_T>     // (res_T::size * 2 == src_T::size)
void split2(hls::stream<src_T> (&input_stream)  [CONFIG_T::n_chan*2/src_T::size],
            hls::stream<res_T> (&output_stream1)[CONFIG_T::n_chan/res_T::size], 
            hls::stream<res_T> (&output_stream2)[CONFIG_T::n_chan/res_T::size]) {
SplitLoop:
    for (int i = 0; i < CONFIG_T::in_height*CONFIG_T::in_width; i++) {
        #pragma HLS PIPELINE

        typename src_T::value_type in_data  [CONFIG_T::n_chan*3];
        typename res_T::value_type out_data1[CONFIG_T::n_chan];
        typename res_T::value_type out_data2[CONFIG_T::n_chan];
        
    ReadPack:
        for (int j = 0; j < 2*CONFIG_T::n_chan/src_T::size; j++) {
            #pragma HLS UNROLL
            src_T in_data_pack = input_stream[j].read();
            for (int k = 0; k < src_T::size; k++) {
                #pragma HLS UNROLL
                in_data[j*src_T::size+k] = in_data_pack[k];
            }
        }
        
    Split3:
        for (int j = 0; j < CONFIG_T::n_chan/res_T::size; j++) {
            #pragma HLS UNROLL
            for (int k = 0; k < res_T::size; k++) {
                #pragma HLS UNROLL
                out_data1[j*res_T::size+k] = in_data[j*res_T::size+k];
                out_data2[j*res_T::size+k] = in_data[j*res_T::size+k+CONFIG_T::n_chan];
        }
    }

    WritePack:
        for (int j =0; j <  CONFIG_T::n_chan/res_T::size; j++) {
            #pragma HLS UNROLL

            res_T out_data1_pack;
            res_T out_data2_pack;
            for (int k = 0; k < res_T::size; k++) {
                #pragma HLS UNROLL
                out_data1_pack[k] = out_data1[j*res_T::size+k];
                out_data2_pack[k] = out_data2[j*res_T::size+k];
            }
            output_stream1[j].write(out_data1_pack);
            output_stream2[j].write(out_data2_pack);
        }
    }
}


// add from 2 input, , from nnet_merge_stream.h::add3
template <class STREAM_T, typename CONFIG_T>
void add2(hls::stream<STREAM_T> (&data1) [CONFIG_T::n_filt/STREAM_T::size], 
          hls::stream<STREAM_T> (&data2) [CONFIG_T::n_filt/STREAM_T::size], 
          hls::stream<STREAM_T> (&res)   [CONFIG_T::n_filt/STREAM_T::size]) {

AddLoop:
    for (int i = 0; i <  CONFIG_T::out_height*CONFIG_T::out_width; i++) {
        #pragma HLS PIPELINE

        typename STREAM_T::value_type in_data1[CONFIG_T::n_filt];
        typename STREAM_T::value_type in_data2[CONFIG_T::n_filt];
        typename STREAM_T::value_type out_data[CONFIG_T::n_filt];
    
    ReadPack:
        for (int j = 0; j < CONFIG_T::n_filt/STREAM_T::size; j++) {
            #pragma HLS UNROLL
            STREAM_T in_data1_pack = data1[j].read();
            STREAM_T in_data2_pack = data2[j].read();
            for (int k = 0; k < STREAM_T::size; k++) {
                #pragma HLS UNROLL
                in_data1[j*STREAM_T::size + k] = in_data1_pack[k];
                in_data2[j*STREAM_T::size + k] = in_data2_pack[k];
            }
        }

    Add:
        for (int j = 0; j < CONFIG_T::n_filt; j++) {
            #pragma HLS UNROLL
            out_data[j] = in_data1[j] + in_data2[j];
        }

    WritePack:
        for (int j = 0; j < CONFIG_T::n_filt/STREAM_T::size; j++) {
            #pragma HLS UNROLL
            STREAM_T out_data_pack;
            for (int k = 0; k < STREAM_T::size; k++) {
                #pragma HLS UNROLL
                out_data_pack[k] = out_data[j*STREAM_T::size + k];
            }
            res[j].write(out_data_pack);
        }
    }
}











// template <class STREAM_T, typename CONFIG_T> 
// void split4(hls::stream<STREAM_T> (&input_stream)[CONFIG_T::n_chan*4], hls::stream<STREAM_T> (&output_stream1)[CONFIG_T::n_chan], hls::stream<STREAM_T> (&output_stream2)[CONFIG_T::n_chan], hls::stream<STREAM_T> (&output_stream3)[CONFIG_T::n_chan], hls::stream<STREAM_T> (&output_stream4)[CONFIG_T::n_chan]) {
//     for (int i = 0; i < CONFIG_T::in_height*CONFIG_T::in_width; i++) {
//         #pragma HLS PIPELINE
//         for (int j = 0; j < CONFIG_T::n_chan; j++) {
//             #pragma HLS UNROLL
//             output_stream1[j].write(input_stream[j].read());
//             output_stream2[j].write(input_stream[j+CONFIG_T::n_chan].read());
//             output_stream3[j].write(input_stream[j+CONFIG_T::n_chan*2].read());
//             output_stream4[j].write(input_stream[j+CONFIG_T::n_chan*3].read());
//         }
//     }
// }

// template <class STREAM_T, typename CONFIG_T> 
// void merge4(hls::stream<STREAM_T> (&input_stream1)[CONFIG_T::n_filt], hls::stream<STREAM_T> (&input_stream2)[CONFIG_T::n_filt], hls::stream<STREAM_T> (&input_stream3)[CONFIG_T::n_filt], hls::stream<STREAM_T> (&input_stream4)[CONFIG_T::n_filt], hls::stream<STREAM_T> (&output_stream)[CONFIG_T::n_filt]) {
//     for (int i = 0; i < CONFIG_T::out_height*CONFIG_T::out_width; i++) {
//         #pragma HLS PIPELINE
//         for (int j = 0; j < CONFIG_T::n_filt; j++) {
//             #pragma HLS UNROLL
//             output_stream[j].write(input_stream1[j].read()+input_stream2[j].read()+input_stream3[j].read()+input_stream4[j].read());
//         }
//     }
// }



















// template <class STREAM_T, typename CONFIG_T> 
// void concat2(hls::stream<STREAM_T> (&input_stream1)[CONFIG_T::n_filt], hls::stream<STREAM_T> (&input_stream2)[CONFIG_T::n_filt], hls::stream<STREAM_T> (&output_stream)[CONFIG_T::n_filt*2]) {
//     for (int i = 0; i < CONFIG_T::out_height*CONFIG_T::out_width; i++) {
//         #pragma HLS PIPELINE
//         for (int j = 0; j < CONFIG_T::n_filt; j++) {
//             #pragma HLS UNROLL
//             output_stream[j].write(input_stream1[j].read());
//             output_stream[j + CONFIG_T::n_filt].write(input_stream2[j].read());
//         }
//     }
// }

// template <class STREAM_T, typename CONFIG_T> 
// void concat4(hls::stream<STREAM_T> (&input_stream1)[CONFIG_T::n_filt], hls::stream<STREAM_T> (&input_stream2)[CONFIG_T::n_filt], hls::stream<STREAM_T> (&input_stream3)[CONFIG_T::n_filt], hls::stream<STREAM_T> (&input_stream4)[CONFIG_T::n_filt], hls::stream<STREAM_T> (&output_stream)[CONFIG_T::n_filt*4]) {
//     for (int i = 0; i < CONFIG_T::out_height*CONFIG_T::out_width; i++) {
//         #pragma HLS PIPELINE
//         for (int j = 0; j < CONFIG_T::n_filt; j++) {
//             #pragma HLS UNROLL
//             output_stream[j].write(input_stream1[j].read());
//             output_stream[j + CONFIG_T::n_filt].write(input_stream2[j].read());
//             output_stream[j + CONFIG_T::n_filt*2].write(input_stream3[j].read());
//             output_stream[j + CONFIG_T::n_filt*3].write(input_stream4[j].read());
//         }
//     }
// }

// template <class STREAM_T, typename CONFIG_T> 
// void split4dense(hls::stream<STREAM_T> (&input_stream)[CONFIG_T::n_in*4], hls::stream<STREAM_T> (&output_stream1)[CONFIG_T::n_in], hls::stream<STREAM_T> (&output_stream2)[CONFIG_T::n_in], hls::stream<STREAM_T> (&output_stream3)[CONFIG_T::n_in], hls::stream<STREAM_T> (&output_stream4)[CONFIG_T::n_in]) {
//     for (int j = 0; j < CONFIG_T::n_in; j++) {
//         #pragma HLS UNROLL
//         output_stream1[j].write(input_stream[j].read());
//         output_stream2[j].write(input_stream[j+CONFIG_T::n_in].read());
//         output_stream3[j].write(input_stream[j+CONFIG_T::n_in*2].read());
//         output_stream4[j].write(input_stream[j+CONFIG_T::n_in*3].read());
//     }
// }

// template <class STREAM_T, typename CONFIG_T> 
// void merge4dense(hls::stream<STREAM_T> (&input_stream1)[CONFIG_T::n_out], hls::stream<STREAM_T> (&input_stream2)[CONFIG_T::n_out], hls::stream<STREAM_T> (&input_stream3)[CONFIG_T::n_out], hls::stream<STREAM_T> (&input_stream4)[CONFIG_T::n_out], hls::stream<STREAM_T> (&output_stream)[CONFIG_T::n_out]) {
//     for (int j = 0; j < CONFIG_T::n_out; j++) {
//         #pragma HLS UNROLL
//         output_stream[j].write(input_stream1[j].read()+input_stream2[j].read()+input_stream3[j].read()+input_stream4[j].read());
//     }
// }

// template <class STREAM_T, typename CONFIG_T> 
// void concat4dense(hls::stream<STREAM_T> (&input_stream1)[CONFIG_T::n_out], hls::stream<STREAM_T> (&input_stream2)[CONFIG_T::n_out], hls::stream<STREAM_T> (&input_stream3)[CONFIG_T::n_out], hls::stream<STREAM_T> (&input_stream4)[CONFIG_T::n_out], hls::stream<STREAM_T> (&output_stream)[CONFIG_T::n_out*4]) {
//     for (int j = 0; j < CONFIG_T::n_out; j++) {
//         #pragma HLS UNROLL
//         output_stream[j].write(input_stream1[j].read());
//         output_stream[j + CONFIG_T::n_out].write(input_stream2[j].read());
//         output_stream[j + CONFIG_T::n_out*2].write(input_stream3[j].read());
//         output_stream[j + CONFIG_T::n_out*3].write(input_stream4[j].read());
//     }
// }

// template <class STREAM_T, typename CONFIG_T> 
// void concat2dense(hls::stream<STREAM_T> (&input_stream1)[CONFIG_T::n_out], hls::stream<STREAM_T> (&input_stream2)[CONFIG_T::n_out], hls::stream<STREAM_T> (&output_stream)[CONFIG_T::n_out*2]) {
//     for (int j = 0; j < CONFIG_T::n_out; j++) {
//         #pragma HLS UNROLL
//         output_stream[j].write(input_stream1[j].read());
//         output_stream[j + CONFIG_T::n_out].write(input_stream2[j].read());
//     }
// }










template <class dataType, unsigned int nrows> int read_file_1D(const char *filename, dataType data[nrows]) {
    FILE *fp;
    fp = fopen(filename, "r");
    if (fp == 0) {
        return -1;
    }
    // Read data from file
    float newval;
    for (int ii = 0; ii < nrows; ii++) {
        if (fscanf(fp, "%f\n", &newval) != 0) {
            data[ii] = newval;
        } else {
            return -2;
        }
    }
    fclose(fp);
    return 0;
}

template <class dataType, unsigned int nrows, unsigned int ncols>
int read_file_2D(const char *filename, dataType data[nrows][ncols]) {
    FILE *fp;
    fp = fopen(filename, "r");
    if (fp == 0) {
        return -1;
    }
    // Read data from file
    float newval;
    for (int ii = 0; ii < nrows; ii++) {
        for (int jj = 0; jj < ncols; jj++) {
            if (fscanf(fp, "%f\n", &newval) != 0) {
                data[ii][jj] = newval;
            } else {
                return -2;
            }
        }
    }
    fclose(fp);
    return 0;
}

template <class in_T, class out_T, int N_IN> void change_type(hls::stream<in_T> &in, hls::stream<out_T> &out) {
    in_T datareg;
    hls::stream<out_T> input_trunc;
    for (int ii = 0; ii < N_IN; ii++) {
        out << (out_T)in.read();
    }
}

template <class data_T, int N_IN> void hls_stream_debug(hls::stream<data_T> &data, hls::stream<data_T> &res) {
    data_T datareg;
    for (int ii = 0; ii < N_IN; ii++) {
        datareg = data.read();
        std::cout << "[" << ii << "]: " << datareg << std::endl;
        res << datareg;
    }
}

constexpr int ceillog2(int x) { return (x <= 2) ? 1 : 1 + ceillog2((x + 1) / 2); }

constexpr int floorlog2(int x) { return (x < 2) ? 0 : 1 + floorlog2(x / 2); }

constexpr int pow2(int x) { return x == 0 ? 1 : 2 * pow2(x - 1); }


//! add   该函数和nnet_commnon.h中DIV_ROUNDUP宏定义一样。
constexpr int ceil_div(int numerator, int denominator) {  //! numerator/denominator的上取整
    return ((numerator - 1) / denominator) + 1;
}


} // namespace nnet

#endif
