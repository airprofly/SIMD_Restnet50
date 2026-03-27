/**
File Name: print.h 
Author: airprofly
Email: 3502196756@qq.com
Date: 2025-04-01
Description: This file has realized the following functions:
            - print the conv layer parameters
            - print the conv layer weights
*/



#ifndef _PRINT_H_
#define _PRINT_H_

#include <iostream>

#include"alignPtr.h"

/**
 * @brief print the conv param

 * @param param the conv param pointer
 * @param len the length of the conv param
 * @param name the name of the conv param

 */
void print_conv_param(const aligned_unique_ptr<int>& param,
                     int len,
                     const std::string& name
                    );

/**
 * @brief print the conv weight

 * @param weight the conv weight pointer
 * @param chan_in the input channel
 * @param chan_out the output channel
 * @param kernel the kernel size
 * @param name the name of the conv weight

 */

void print_conv_weight(const aligned_unique_ptr<float>& weight,
                     int chan_in,
                     int chan_out,
                     int kernel,
                     const std::string& name
                    );

#endif