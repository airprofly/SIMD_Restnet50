#include "print.h"
#include "param.h"

/**
 * @brief print the conv param

 * @param param the conv param pointer
 * @param len the length of the conv param

 */
void print_conv_param(const aligned_unique_ptr<int> &param, // the conv param pointer
                      int len,                              // the length of the conv param
                      const std::string &name               // the name of the conv param
)
{
    if (len != CONV_PARAM_SIZE)
    {
        throw std::invalid_argument("the conv param size is not correct");
    }
    std::cout<<name<<std::endl;
    std::cout << std::endl
              << std::endl;
    std::cout << "the input channel is: " << param[0] << std::endl;
    std::cout << "the output channel is: " << param[1] << std::endl;
    std::cout << "the kernel size is: " << param[2] << std::endl;
    std::cout << "the stride is: " << param[3] << std::endl;
    std::cout << "the padding is: " << param[4] << std::endl;
    std::cout << std::endl
              << std::endl
              << std::endl
              << std::endl;

    std::cout << "================================" << std::endl;
}

/**
 * @brief print the conv weight

 * @param weight the conv weight pointer
 * @param chan_in the input channel
 * @param chan_out the output channel
 * @param kernel the kernel size
 * @param name the name of the conv weight

 */
void print_conv_weight(const aligned_unique_ptr<float>& weight, // the conv weight pointer
                       int chan_in,                             // the input channel
                       int chan_out,                            // the output channel
                       int kernel,                              // the kernel size
                       const std::string& name                  // the name of the conv weight
)
{
    std::cout <<name << std::endl;
    std::cout<<"the conv weight size is"<<chan_out<<" * "<<kernel<< " * "<<kernel<< " * "<<chan_in<<std::endl<<std::endl<<std::endl;
    for(int i{0};i<chan_out;i++){
        std::cout<<"this conv weight mat is "<<i<<std::endl<<std::endl;
        for(int j{0};j<kernel;j++){
            for(int k{0};k<kernel;k++){
                for(int ci{0};ci<chan_in;ci++){
                    std::cout<<weight[i*kernel*kernel*chan_in+j*kernel*chan_in+k*chan_in+ci]<<" ";
                }
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }
    std::cout << "================================" << std::endl;
}