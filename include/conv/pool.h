/**
File Name: pool.h
Author: airprofly
Email: 3502196756@qq.com
Date: 2025-04-02
Description: This file has realized the following functions:
            - maxpool function
            - avgpool function
*/
#include "alignPtr.h"
#include "param.h"

#ifndef _POOL_H_
#define _POOL_H_

/**
 * @brief maxpool function

 * @param chan_in the channel of the input feature mat
 * @param hei_in the height of the input feature mat
 * @param wid_in the width of the input feature mat
 * @param hei_out the height of the output feature mat
 * @param wid_out the width of the output feature mat
 * @param kernel the kernel size
 * @param stride the stride size
 * @param pad the padding size

 * @return the maxpooled feature mat

 * @note the input feature will be freed
 */
template <typename T>
aligned_unique_ptr<T> maxPool_layer(aligned_unique_ptr<T> &input, // the input feature mat
                                    int chan_in,                  // the channel of the input feature mat
                                    int hei_in,                   // the height of the input feature mat
                                    int wid_in,                   // the width of the input feature mat
                                    int &hei_out,                 // the height of the output feature mat
                                    int &wid_out,                 // the width of the output feature mat
                                    int kernel,                   // the kernel size
                                    int stride,                   // the stride size
                                    int pad                       // the padding size
)
{
    /* calculate the height and width of the output feature mat */
    hei_out = (hei_in + 2 * pad - kernel) / stride + 1;
    wid_out = (wid_in + 2 * pad - kernel) / stride + 1;

    aligned_unique_ptr<T> output = make_aligned_unique<T>(hei_out * wid_out * chan_in, ALIGN_SIZE); // malloc aligned memory

    for (int out_chan_idx{0}; out_chan_idx < chan_in; ++out_chan_idx)
    {
        for (int hei_out_idx{0}; hei_out_idx < hei_out; hei_out_idx++)
        {
            const int in_hei_start = hei_out_idx * stride - pad;
            for (int wid_out_idx{0}; wid_out_idx < wid_out; wid_out_idx++)
            {
                const int in_wid_start = wid_out_idx * stride - pad;
                // (in_hei_start,in_wid_start) is the left top start point of the input feature mat

                const int kenerl_hei_start = std::max(0, -in_hei_start);
                const int kenerl_wid_start = std::max(0, -in_wid_start);
                const int kenerl_hei_end = std::min(kernel, hei_in - in_hei_start);
                const int kenerl_wid_end = std::min(kernel, wid_in - in_wid_start);

                float maxx{0}; // initialize the max value,because the relu_layer will set the min value to 0

                // use the kernel to maxpool the temp postion of the input feature mat
                for (int kenerl_hei_idx{kenerl_hei_start}; kenerl_hei_idx < kenerl_hei_end; ++kenerl_hei_idx)
                {
                    const int in_hei_idx = in_hei_start + kenerl_hei_idx;
                    for (int kenerl_wid_idx{kenerl_wid_start}; kenerl_wid_idx < kenerl_wid_end; ++kenerl_wid_idx)
                    {
                        const int in_wid_idx = in_wid_start + kenerl_wid_idx;

                        T in_data = input[in_hei_idx * wid_in * chan_in + in_wid_idx * chan_in + out_chan_idx];
                        maxx = std::max(in_data, maxx); // find the max value in the kernel
                    }
                }
                output[hei_out_idx * wid_out * chan_in + wid_out_idx * chan_in + out_chan_idx] = maxx;
            }
        }
    }
    input.reset();
    return std::move(output); // return the maxpooled feature mat
}

/**
 * @brief average pool function

 * @param chan_in the channel of the input feature mat
 * @param hei_in the height of the input feature mat
 * @param wid_in the width of the input feature mat
 * @param hei_out the height of the output feature mat
 * @param wid_out the width of the output feature mat
 * @param kernel the kernel size
 * @param stride the stride size
 * @param pad the padding size

 * @return the average pooled feature mat

 * @note the input feature will be freed
 */
template <typename T>
aligned_unique_ptr<T> avgPool_layer(aligned_unique_ptr<T> &input, // the input feature mat
                                    int chan_in,                  // the channel of the input feature mat
                                    int hei_in,                   // the height of the input feature mat
                                    int wid_in,                   // the width of the input feature mat
                                    int &hei_out,                 // the height of the output feature mat
                                    int &wid_out,                 // the width of the output feature mat
                                    int kernel,                   // the kernel size
                                    int stride,                   // the stride size
                                    int pad                       // the padding size
)
{
    /* calculate the height and width of the output feature mat */
    hei_out = (hei_in + 2 * pad - kernel) / stride + 1;
    wid_out = (wid_in + 2 * pad - kernel) / stride + 1;

    aligned_unique_ptr<T> output = make_aligned_unique<T>(hei_out * wid_out * chan_in, ALIGN_SIZE); // malloc aligned memory

    for (int out_chan_idx{0}; out_chan_idx < chan_in; ++out_chan_idx)
    {
        for (int hei_out_idx{0}; hei_out_idx < hei_out; hei_out_idx++)
        {
            const int in_hei_start = hei_out_idx * stride - pad;
            for (int wid_out_idx{0}; wid_out_idx < wid_out; wid_out_idx++)
            {
                const int in_wid_start = wid_out_idx * stride - pad;
                // (in_hei_start,in_wid_start) is the left top start point of the input feature mat

                const int kenerl_hei_start = std::max(0, -in_hei_start);
                const int kenerl_wid_start = std::max(0, -in_wid_start);
                const int kenerl_hei_end = std::min(kernel, hei_in - in_hei_start);
                const int kenerl_wid_end = std::min(kernel, wid_in - in_wid_start);

                float sum{0}; // initialize the max value,because the relu_layer will set the min value to 0

                // use the kernel to maxpool the temp postion of the input feature mat
                for (int kenerl_hei_idx{kenerl_hei_start}; kenerl_hei_idx < kenerl_hei_end; ++kenerl_hei_idx)
                {
                    const int in_hei_idx = in_hei_start + kenerl_hei_idx;
                    for (int kenerl_wid_idx{kenerl_wid_start}; kenerl_wid_idx < kenerl_wid_end; ++kenerl_wid_idx)
                    {
                        const int in_wid_idx = in_wid_start + kenerl_wid_idx;

                        T in_data = input[in_hei_idx * wid_in * chan_in + in_wid_idx * chan_in + out_chan_idx];
                        sum += in_data; // find the max value in the kernel
                    }
                }
                output[hei_out_idx * wid_out * chan_in + wid_out_idx * chan_in + out_chan_idx] = sum / (kernel * kernel); // average the sum
            }
        }
    }
    input.reset();
    return std::move(output); // return the maxpooled feature mat
}

#endif
