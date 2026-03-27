#include "matrix.h"

#include <immintrin.h>
#include <iostream>

#include "param.h"


/**
 * @brief calculate the 2d convolution result
 *
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

 *
 * @return the output feature matrix
 *
 * @note the feacure mat is stored in format [hei][wid][chan]
 * @note the weight mat is stored in format [N][hei][wid][chan],just NHWC format
 * @note the output mat is stored in format [hei][wid][chan]
 */
aligned_unique_ptr<float> conv2d(aligned_unique_ptr<float>& input,  // the input feature mat
                                 aligned_unique_ptr<float>& weight, // the weight matrix
                                 int hei_in,                        // the height of the input feature mat
                                 int wid_in,                        // the width of the input feature mat
                                 int chan_in,                       // the channel of the input feature mat
                                 int& hei_out,                      // the height of the output feature mat
                                 int& wid_out,                      // the width of the output feature mat
                                 int chan_out,                      // the channel of the output feature mat
                                 int kernel,                        // the kernel size
                                 int stride,                        // the stride
                                 int pad,                           // the padding
                                 bool is_first,
                                 bool is_free_input // the flag to indicate whether to free the input feature mat
)
{
    /* calculate the output feature mat size */
    hei_out = (hei_in + 2 * pad - kernel) / stride + 1;
    wid_out = (wid_in + 2 * pad - kernel) / stride + 1;

    aligned_unique_ptr<float> output = make_aligned_unique<float>(hei_out * wid_out * chan_out, ALIGN_SIZE); // the memory to store the output feature mat

    for (int out_chan_idx{0}; out_chan_idx < chan_out; ++out_chan_idx) // the index of the output feature mat
    {
        for (int out_hei_idx{0}; out_hei_idx < hei_out; ++out_hei_idx) // the row index of the current output feature mat
        {
            const int input_hei_start = out_hei_idx * stride - pad;        // the start row index of the current output feature mat
            for (int out_wid_idx{0}; out_wid_idx < wid_out; ++out_wid_idx) // the column index of the current feature mat
            {
                const int input_wid_start = out_wid_idx * stride - pad; // the start column index of the current feature mat
                // the (input_hei_start, input_wid_start) is the left top corner of the current kernel corresponding to the current output feature mat positon

                const int kernel_hei_start = std::max(0, -input_hei_start); // the start row index of the current kernel
                const int kernel_wid_start = std::max(0, -input_wid_start); // the start column index of the current kernel
                // if the postion contain the padding part,becase the kenel using the 0 padding
                // so the calulation of the padding part is not needed,
                // then we change the start and end index of the kernel to the useful calulation part

                const int kernel_hei_end = std::min(kernel, hei_in - input_hei_start); // calu the end idx of the feature mat
                const int kernel_wid_end = std::min(kernel, wid_in - input_wid_start);

                /*the postion has fixed,then multiply the mat*/

                float sum{0};
                if (is_first) // if the current conv is the first conv
                {
                    for (int kernel_hei_idx{kernel_hei_start}; kernel_hei_idx < kernel_hei_end; kernel_hei_idx++) {
                        const int input_hei_idx = input_hei_start + kernel_hei_idx; // the current  row of the feature mat
                        for (int kernel_wid_idx{kernel_wid_start}; kernel_wid_idx < kernel_wid_end; kernel_wid_idx++) {
                            const int input_wid_idx = input_wid_start + kernel_wid_idx; // the current  col of the feature mat
                            for (int ci_ = 0; ci_ < 3; ci_++) {
                                const float input_data = input[input_hei_idx * IMG_COL_SIZE * 3 + input_wid_idx * 3 + ci_];
                                const float weight_data = weight[out_chan_idx * 49 * 3 + kernel_hei_idx * 7 * 3 + kernel_wid_idx * 3 + ci_];
                                sum += input_data * weight_data;
                            }
                        }

                    } // complete once kenel convelution, just at a postion
                }
                else // if not the first convelution,use the avx2 to accelerate the calculation
                {
                    const int vec_size = 8; // the register can store 8 float data
                    if (chan_in % vec_size != 0) {
                        throw std::runtime_error("the channel of the input feature mat is not a multiple of 8");
                        return nullptr;
                    }
                    for (int kernel_hei_idx{kernel_hei_start}; kernel_hei_idx < kernel_hei_end; kernel_hei_idx++) {
                        const int input_hei_idx = input_hei_start + kernel_hei_idx;
                        for (int kernel_wid_idx{kernel_wid_start}; kernel_wid_idx < kernel_wid_end; kernel_wid_idx++) {
                            const int input_wid_idx = input_wid_start + kernel_wid_idx;
                            __m256 in_vec, weight_vec;        // 256bit register,store 8 float data.the input feature mat channels and weight mat channels
                            __m256 acc = _mm256_setzero_ps(); // the register to store the result
                            for (int ci_{0}; ci_ < chan_in; ci_ += vec_size) {
                                in_vec = _mm256_load_ps(&input[input_hei_idx * wid_in * chan_in + input_wid_idx * chan_in + ci_]);
                                weight_vec = _mm256_load_ps(&weight[out_chan_idx * kernel * kernel * chan_in + kernel_hei_idx * kernel * chan_in + kernel_wid_idx * chan_in + ci_]);

                                acc = _mm256_fmadd_ps(in_vec, weight_vec, acc);
                            }
                            float* acc_ptr = (float*)&acc;
                            for (int i = 0; i < vec_size; i++) { sum += acc_ptr[i]; }
                        }
                    }
                }

                output[out_hei_idx * wid_out * chan_out + out_wid_idx * chan_out + out_chan_idx] = sum; // store the result
            }
        }
    }
    weight.reset();
    if (is_free_input) { input.reset(); }
    return std::move(output); // return the result
}
