/**
File Name: conv.h 
Author: airprofly
Email: 3502196756@qq.com
Date: 2025-04-01
Description: This file has completed the following functions:
                - defined the convolution layer
                - load the conv layer parameters、weights
                - the relu function
*/


#ifndef _CONV_H_
#define _CONV_H_

#include <filesystem>

#include "param.h"
#include "file_util.h"

/**
 * @brief compute the convolution layer

 * @param input the input feature mat
 * @param hei_in the height of the input feature mat
 * @param wid_in the width of the input feature mat
 * @param hei_out the height of the output feature mat
 * @param wid_out the width of the output feature mat
 * @param chan_out the channel of the output feature mat
 * @param layer_name the name of the layer
 * @param is_free_input the flag to indicate whether the input feature mat should be freed

 * @return the output feature mat pointer
 * @note the output mat is stored in format [hei_out, wid_out, chan_out],just HWC format
 */
aligned_unique_ptr<float> compute_conv_layer(aligned_unique_ptr<float>& input, // the input feature mat
                                             int hei_in,                      // the height of the input feature mat
                                             int wid_in,                      // the width of the input feature mat
                                             int &hei_out,                    // the height of the output feature mat
                                             int &wid_out,                    // the width of the output feature mat
                                             int &chan_out,                   // the channel of the output feature mat
                                             const std::string &layer_name,   // the name of the layer
                                             bool is_free_input = true        // the flag to indicate whether the input feature mat should be freed
);




/**
 * @brief load the convolution parameters from the file

 * @param path the path of to the convolution layer param txt
 * @param cnt the number of parameters to be loaded

 * @return the pointer of the parameters
 */
inline aligned_unique_ptr<int> load_conv_param(const std::filesystem::path path, const int cnt)
{
    return std::move(load_data_from_file<int>(path, cnt));
}

/**
 * @brief load the convolution weight from the file

 * @param path the path of to the convolution layer weight txt
 * @param cnt the number of weight to be loaded

 * @return the pointer of the weight
 */
inline aligned_unique_ptr<float> load_conv_weight(const std::filesystem::path path, const int cnt)
{
    return std::move(load_data_from_file<float>(path, cnt));
}



#endif