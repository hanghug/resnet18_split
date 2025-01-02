# GALA_PATH  = ../galapagos
GALA_PATH = $(GALAPAGOS_PATH)

# Compiler
CXX = g++

INCLUDE_FLAGS = -I$(GALA_PATH)/middleware/libGalapagos \
				-I$(CURDIR)/firmware/nnet_utils  \
				-I$(CURDIR)/firmware/ap_types  \
				-I$(CURDIR)/firmware  \
				-I$(CURDIR)/kernel \
				-I$(HLS_PATH) \
				-I$(GALA_PATH)/util/spdlog/include \
				-I$(GALA_PATH)/util/Catch2/single_include/catch2 
				
		          
#CXXFLAGS = -DCPU -DLOG_LEVEL=2 -O2 -std=c++17 -pthread $(INCLUDE_FLAGS)   #!debug级别为2,不需要时改为0，LOG_LEVEL=0。
CXXFLAGS = -DRECV_SIM -O2 -std=c++17 -pthread $(INCLUDE_FLAGS)
#CXXFLAGS = -DINTERFACE_100G -DRECV_SIM -O2 -std=c++17 -pthread $(INCLUDE_FLAGS)
# For 512 data width

BOOST_LDFLAGS = -lboost_thread -lboost_system -lpthread

MODE_TEST_FILES = resnet18_split_test.cpp    #!模型测试函数
MODE_TEST_OBJ = $(MODE_TEST_FILES:.cpp=.o)   #! *_test.o
MODE_TEST_TARGET = $(MODE_TEST_FILES:.cpp=)  #! *_test

MODE_FILES = ./firmware/resnet18_split.cpp   #!模型名称.cpp
MODE_OBJ = $(MODE_FILES:.cpp=.o)

KERNEL_TEST_FILES = resnet18_kernel_test.cpp    #!模型测试函数
KERNEL_TEST_OBJ = $(KERNEL_TEST_FILES:.cpp=.o)   
KERNEL_TEST_TARGET = $(KERNEL_TEST_FILES:.cpp=)  

KERNEL_FILES = ./kernel/resnet18_kernel.cpp  #! kerenl名称.cpp
KERNEL_OBJ = $(KERNEL_FILES:.cpp=.o)


all: $(MODE_TEST_TARGET)

$(KERNEL_TEST_TARGET): $(MODE_OBJ)  $(KERNEL_OBJ) $(KERNEL_TEST_OBJ)
	$(CXX) $^ -o $@ $(CXXFLAGS) $(BOOST_LDFLAGS)
	@./$@

$(MODE_TEST_TARGET): $(MODE_OBJ) $(MODE_TEST_OBJ)
	$(CXX) $^ -o $@ $(CXXFLAGS)   
	@./$@

%.o: %.cpp
	$(CXX) -c $< -o $@  $(CXXFLAGS) $(BOOST_LDFLAGS)  

RPT_DIR = $(foreach i, $(shell seq 7 7),kernel${i}_wrapper_prj/solution1/syn/report/kernel${i}_wrapper_csynth.rpt  kernel${i}_wrapper_prj/solution1/syn/report/kernel${i}_csynth.rpt)

get_rpt:
	@mkdir -p rpt
	@for dir in ${RPT_DIR}; \
	do \
		cp $${dir} rpt/; \
	done

IP_ZIP = $(foreach i, $(shell seq 7 7),kernel${i}_wrapper_prj/solution1/impl/ip/xilinx_com_hls_kernel${i}_wrapper_1_0.zip)

get_ip:
	@mkdir -p ip
	@for zip in ${IP_ZIP}; \
	do \
		cp $${zip} ip/; \
	done

# insert syn tcl     make syn -jnum  (num为进程数)可实现并行执行
syn: k7 k9 k10
send:
	vitis_hls -f build_prj_split.tcl index=0 | tee s0.log
k1:
	vitis_hls -f build_prj_split.tcl top=kernel1_wrapper prefix=k1_ index=1 > k1.log
k2:
	vitis_hls -f build_prj_split.tcl top=kernel2_wrapper prefix=k2_ index=2 > k2.log
k3:
	vitis_hls -f build_prj_split.tcl top=kernel3_wrapper prefix=k3_ index=3 > k3.log
k4:
	vitis_hls -f build_prj_split.tcl top=kernel4_wrapper prefix=k4_ index=4 > k4.log
k5:
	vitis_hls -f build_prj_split.tcl top=kernel5_wrapper prefix=k5_ index=5 > k5.log
k6:
	vitis_hls -f build_prj_split.tcl top=kernel6_wrapper prefix=k6_ index=6 > k6.log
k7:
	vitis_hls -f build_prj_split.tcl top=kernel7_wrapper prefix=k7_ index=7 > k7.log
k8:
	vitis_hls -f build_prj_split.tcl top=kernel8_wrapper prefix=k8_ index=8 > k8.log
k9:
	vitis_hls -f build_prj_split.tcl top=kernel9_wrapper prefix=k9_ index=9 > k9.log
k10:
	vitis_hls -f build_prj_split.tcl top=kernel10_wrapper prefix=k10_ index=10 > k10.log
k11:
	vitis_hls -f build_prj_split.tcl top=kernel11_wrapper prefix=k11_ index=11 > k11.log

tar:
	tar -zcvf verilog.tar.gz *_prj/solution1/syn/verilog

clean:
	rm -rf $(MODE_OBJ) $(KERNEL_OBJ) $(MODE_TEST_OBJ) $(KERNEL_TEST_OBJ) $(MODE_TEST_TARGET) $(KERNEL_TEST_TARGET) \
			dataflow_log.txt  kernel_log.txt

delete:
	ls | grep -v "gp_layout_split.yaml" | xargs rm -rf