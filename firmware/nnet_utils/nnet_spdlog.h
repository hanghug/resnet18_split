#ifndef NNET_SPDLOG_H   //!不可综合该库函数
#define NNET_SPDLOG_H

#include <iostream>
#include "hls_stream.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/fmt/bin_to_hex.h"


namespace nnet { 
    
template <class res_T, size_t SIZE>

//! 在log中输出hls流
void spdlog_result(hls::stream<res_T> &result, std::shared_ptr<spdlog::logger> logger, bool keep = false) {

    for (int i = 0; i < SIZE / res_T::size; i++) {
        res_T res_pack = result.read();
        std::string log_message = "";

        for (int j = 0; j < res_T::size; j++) {
            log_message += std::to_string(res_pack[j].to_float()) + " ";  //! 希望输出在一行上
        }
        
        logger->debug(log_message);//!使用 spdlog 输出日志消息

        if (keep)
            result.write(res_pack);
    }

}

}

#endif