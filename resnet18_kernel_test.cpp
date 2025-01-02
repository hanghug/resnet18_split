#include "resnet18_kernel.h"


int main() {

    galapagos_interface input_1;
    //insert variables define
    galapagos_interface layer12_out;
    galapagos_interface layer29_out;
    galapagos_interface layer36_out;
    galapagos_interface layer46_out;
    galapagos_interface layer53_out;
    galapagos_interface layer56_out;
    galapagos_interface layer81_cpy2;
    galapagos_interface layer62_out;
    galapagos_interface layer61_out;
    galapagos_interface layer66_out;
    galapagos_interface layer82_cpy2;
    galapagos_interface layer70_out;
    galapagos_interface layer74_out;
    

    prepare_data_kernel(&input_1);
    //insert functions kernel define
    kernel1_wrapper(&input_1,&layer12_out);
    kernel2_wrapper(&layer12_out,&layer29_out);
    kernel3_wrapper(&layer29_out,&layer36_out);
    kernel4_wrapper(&layer36_out,&layer46_out);
    kernel5_wrapper(&layer46_out,&layer53_out);
    kernel6_wrapper(&layer53_out,&layer56_out,&layer81_cpy2);
    kernel7_wrapper(&layer56_out,&layer62_out);
    kernel8_wrapper(&layer81_cpy2,&layer61_out);
    kernel9_wrapper(&layer61_out,&layer62_out,&layer66_out,&layer82_cpy2);
    kernel10_wrapper(&layer66_out,&layer82_cpy2,&layer70_out);
    kernel11_wrapper(&layer70_out,&layer74_out);

    recv_data_kernel(&layer74_out);
    
    return 0;
}

