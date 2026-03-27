/**
File Name: fc.h
Author: airprofly
Email: 3502196756@qq.com
Date: 2025-04-02
Description: This file has realized the following functions:
              - fc function
*/
#ifndef _FC_H_
#define _FC_H_

#include <immintrin.h>

#include "alignPtr.h"
#include "param.h"

/**
 * @brief fully connected function

 * @param input the input feature mat
 * @param weight the weight mat
 * @param bias the bias mat
 * @param len the length of the feature mat

 * @return the expected score of every class
 */
template <typename T>
aligned_unique_ptr<T> fc(aligned_unique_ptr<T> &input,  // the input feature mat
                         aligned_unique_ptr<T> &weight, // the weight mat
                         aligned_unique_ptr<T> &bias,   // the bias mat
                         int len                        // the length of the feature mat
)
{
    aligned_unique_ptr<T> out = make_aligned_unique<T>(OUTPUT_CLASS_NUM, ALIGN_SIZE); // the divide classes epected score

    for (int i = 0; i < OUTPUT_CLASS_NUM; i++)
    {
        if (len % FLOAT_NUM != 0)
        {
            throw std::runtime_error("the length of the input feature mat is not a multiple of 8 to using SIMD to accelerate the fc layer");
            return nullptr;
        }
        __m256 sum_value = _mm256_setzero_ps();
        for (int j = 0; j < len; j += FLOAT_NUM)
        {
            __m256 input_value = _mm256_load_ps(&input[j]); // load the input feature mat
            __m256 weight_value = _mm256_load_ps(&weight[i * len + j]);

            sum_value = _mm256_fmadd_ps(input_value, weight_value, sum_value);
        }
        float sum{0};
        __m128 low = _mm256_extractf128_ps(sum_value, 0);  // get the low 128 bits of the sum_value
        __m128 high = _mm256_extractf128_ps(sum_value, 1); // get the high 128 bits of the sum_value
        __m128 temp = _mm_add_ps(low, high);               // add the low and high 128 bits
        temp = _mm_hadd_ps(temp, temp);
        temp = _mm_hadd_ps(temp, temp);
        sum = _mm_cvtss_f32(temp); // get the sum value

        out[i] = sum + bias[i]; // the output score
    }

    input.reset();  // release the input feature mat
    weight.reset(); // release the weight mat
    bias.reset();

    return std::move(out);
}

/**
 * @brief fully connected function layer

 * @param input the input feature mat
 * @param layer_name the name of the weight file
 * @param len the length of the feature mat

 * @return the expected score of every class
 */
aligned_unique_ptr<float> fc_layer(aligned_unique_ptr<float> &input, // the input feature mat
                                   const std::string layer_name,     // the name of the weight file
                                   int len                           // the length of the feature mat
);

#endif
