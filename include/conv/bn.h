/**
File Name: bn.h
Author: airprofly
Email: 3502196756@qq.com
Date: 2025-04-01
Description: This file has realized the following functions:
            - defined the batch normalization layer

*/
#include "alignPtr.h"

#ifndef _BN_H_
#define _BN_H_

/**
 * @brief compute the batch normalization layer

 * @param input the input feature mat
 * @param mean the mean of every channel
 * @param var the variance of every channel
 * @param gamma the scale parameter of every channel
 * @param bias the bias parameter of every channel
 * @param hei_in the height of the input feature mat
 * @param wid_in the width of the input feature mat
 * @param chan_in the channel of the input feature mat

 * @return the batch normalization result
 */
aligned_unique_ptr<float> bn(aligned_unique_ptr<float> &input, // the input feature mat
                             aligned_unique_ptr<float> &mean,  // the mean of every channel
                             aligned_unique_ptr<float> &var,   // the variance of every channel
                             aligned_unique_ptr<float> &gamma, // the scale parameter of every channel
                             aligned_unique_ptr<float> &bias,  // the bias parameter of every channel
                             int hei_in,                       // the height of the input feature mat
                             int wid_in,                       // the width of the input feature mat
                             int chan_in                       // the channel of the input feature mat
);



/**
 * @brief compute the batch normalization layer with para from file name provided

 * @param input the input feature mat
 * @param hei_in the height of the input feature mat
 * @param wid_in the width of the input feature mat
 * @param chan_in the channel of the input feature mat
 * @param layer_name the name of the layer

 * @return the batch normalization result
 */
aligned_unique_ptr<float> bn_layer(aligned_unique_ptr<float> &input, // the input feature mat
                                   int hei_in,                       // the height of the input feature mat
                                   int wid_in,                       // the width of the input feature mat
                                   int chan_in,                      // the channel of the input feature mat
                                   const std::string &layer_name     // the name of the layer
);

#endif