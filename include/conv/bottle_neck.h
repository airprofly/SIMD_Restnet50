/**
File Name: bottle_neck.h
Author: airprofly
Email: 3502196756@qq.com
Date: 2025-04-02
Description: This file has realized the following functions:
            - compute the bottleneck layer(BTNK)
*/
#ifndef _BOTTLE_NECK_H_
#define _BOTTLE_NECK_H_

#include "alignPtr.h"

/**
 * @brief compute the bottleneck layer(BTNK)

 * @param input the input feature mat
 * @param hei_in the height of the input feature mat
 * @param wid_in the width of the input feature mat
 * @param hei_out the height of the output feature mat
 * @param wid_out the width of the output feature mat
 * @param chan_out the channel of the output feature mat
 * @param layer_name the name of the layer
 * @param is_down_sample the flag to indicate whether the input feature mat should be downsampled

 * @return the bottleneck layer output feature mat
 */
aligned_unique_ptr<float> bottleNeck_layer(aligned_unique_ptr<float> &input,         // the input feature mat
                                           int hei_in,                               // the height of the input feature mat
                                           int wid_in,                               // the width of the input feature mat
                                           int &hei_out,                             // the height of the output feature mat
                                           int &wid_out,                             // the width of the output feature mat
                                           int &chan_out,                            // the channel of the output feature mat
                                           const std::string &bottleNeck_layer_name, // the name of the layer
                                           bool is_down_sample                       // the flag to indicate whether the input feature mat should be downsampled
);


/**
 * @brief compute the add operation of the bottleneck layer(BTNK)

 * @param original the original feature  mat without convelution,batch norm and activation
 * @param conv_out the convelutioned feature mat
 * @param len the length of the feature mat

 * @return the bottleneck layer output feature mat
 * @note the conv_out mat will be freed
 */

aligned_unique_ptr<float> BTNK_add(aligned_unique_ptr<float> &original, // the original feature  mat without convelution,batch norm and activation
                                   aligned_unique_ptr<float> &conv_out, // the convelutioned feature mat
                                   int len                              // the length of the feature mat
);

#endif