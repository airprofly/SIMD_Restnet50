/**
File Name: matrix.h 
Author: airprofly
Email: 3502196756@qq.com
Date: 2025-04-01
Description: This file has realized the following functions:
            - convelution the two matrix with the SIMD -avx2 acceleration
*/



#ifndef _MATRIX_H_
#define _MATRIX_H_

#include "alignPtr.h"

/**
 * @brief calculate the 2d convolution result

 * @param input the input feature matrix
 * @param weight the weight matrix
 * @param hei_in the height of the input feature matrix
 * @param wid_in the width of the input feature matrix
 * @param chan_in the channel of the input feature matrix
 * @param hei_out the height of the output feature matrix
 * @param wid_out the width of the output feature matrix
 * @param chan_out the channel of the output feature matrix
 * @param kernel the kernel size
 * @param stride the stride
 * @param pad the padding
 * @param is_first the flag to indicate whether the current conv is the first conv
 * @param is_free_input the flag to indicate whether the input feature matrix should be freed

 * @return
 */
aligned_unique_ptr<float> conv2d(aligned_unique_ptr<float>& input,  // the input feature mat
                                 aligned_unique_ptr<float>& weight, // the weight matrix
                                 int hei_in,                       // the height of the input feature mat
                                 int wid_in,                       // the width of the input feature mat
                                 int chan_in,                      // the channel of the input feature mat
                                 int &hei_out,                     // the height of the output feature mat
                                 int &wid_out,                     // the width of the output feature mat
                                 int chan_out,                     // the channel of the output feature mat
                                 int kernel,                       // the kernel size
                                 int stride,                       // the stride
                                 int pad,                          // the padding
                                 bool is_first,             // the flag to indicate whether the current conv is the first conv
                                 bool is_free_input = true         // the flag to indicate whether the input feature mat should be freed
);

#endif