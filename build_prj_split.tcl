#################
#!    HLS4ML
#! 需要对 resnet18、select_part、select_clock、select_uncertainty进行替换
#################
array set opt {
    reset      1
    csim       0
    synth      1
    cosim      0
    validation 0
    export     1
    vsynth     0
    fifo_opt   0
    model_name "resnet18"
    top        "send"
    prefix     "s0_"
    index      0
}

foreach arg $::argv {
    foreach o [lsort [array names opt]] {
        regexp "$o=+(\\w+)" $arg unused opt($o)
    }
}


set model_name $opt(model_name)
set top $opt(top)
set prefix $opt(prefix)
set index $opt(index)
set model_file_name "${model_name}_split${index}.cpp"
set kernel_file_name "${model_name}_kernel${index}.cpp"


proc report_time { op_name time_start time_end } {
    set time_taken [expr $time_end - $time_start]
    set time_s [expr ($time_taken / 1000) % 60]
    set time_m [expr ($time_taken / (1000*60)) % 60]
    set time_h [expr ($time_taken / (1000*60*60)) % 24]
    puts "***** ${op_name} COMPLETED IN ${time_h}h${time_m}m${time_s}s *****"
}

# set GALA_PATH "../galapagos"
set GALA_PATH $env(GALAPAGOS_PATH)
set src_path_root [pwd]

# set INCLUDE_FLAGS "-I$src_path_root/firmware  -I$src_path_root/firmware/syn_subSplit  \
#                     -I$src_path_root/kernel \
#                    -I$GALA_PATH/middleware/libGalapagos \
#                    -I$src_path_root/firmware/nnet_utils -std=c++0x" 

set INCLUDE_FLAGS "-DINTERFACE_100G -I$src_path_root/firmware  -I$src_path_root/firmware/syn_subSplit  \
                    -I$src_path_root/kernel \
                   -I$GALA_PATH/middleware/libGalapagos \
                   -I$src_path_root/firmware/nnet_utils -std=c++0x" 
# for 512 data width

if {$opt(reset)} {
    open_project -reset ${top}_prj
} else {
    open_project ${top}_prj
}
set_top ${top}
if {$index == 0} {
    add_files kernel/syn_subKernel/${model_name}_send.cpp -cflags  $INCLUDE_FLAGS
} else {
    add_files firmware/syn_subSplit/$model_file_name -cflags  $INCLUDE_FLAGS
    add_files kernel/syn_subKernel/$kernel_file_name -cflags $INCLUDE_FLAGS
}


if {$opt(reset)} {
    open_solution -reset "solution1"  -flow_target vivado
} else {
    open_solution "solution1"  -flow_target vivado
}

#! 
config_rtl -module_prefix ${prefix} 
# config_schedule -enable_dsp_full_reg=false 
# catch {config_array_partition -maximum_size 4096}
# config_compile -name_max_length 80

set_part {xcvu19p-fsva3824-1-e}   
# xcvu19p-fsva3824-1-e
config_schedule -enable_dsp_full_reg=false
create_clock -period 10 -name default   
# set_clock_uncertainty 12.5% default  


if {$opt(csim)} {
    puts "***** C SIMULATION *****"
    set time_start [clock clicks -milliseconds]
    csim_design
    set time_end [clock clicks -milliseconds]
    report_time "C SIMULATION" $time_start $time_end
}

if {$opt(synth)} {
    puts "***** C/RTL SYNTHESIS *****"
    set time_start [clock clicks -milliseconds]
    csynth_design
    set time_end [clock clicks -milliseconds]
    report_time "C/RTL SYNTHESIS" $time_start $time_end
}

if {$opt(export)} {
    puts "***** EXPORT IP *****"
    set time_start [clock clicks -milliseconds]
    export_design -format ip_catalog
    set time_end [clock clicks -milliseconds]
    report_time "EXPORT IP" $time_start $time_end
}


exit
