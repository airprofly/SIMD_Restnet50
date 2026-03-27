/**
File Name: relu.h 
Author: airprofly
Email: 3502196756@qq.com
Date: 2025-04-02
Description: This file has realized the following functions:
            - relu function
*/


#ifndef _RELU_H_
#define _RELU_H_

#include"alignPtr.h"

/**
 * @brief relu function
 * when the value is less than 0, set it to 0;
 * else keep it unchanged

 * @param mat the pointer of the data
 * @param len the length of the data

 * @return the pointer of the data
 */
template <typename T>
aligned_unique_ptr<T> relu_layer(aligned_unique_ptr<T>& mat, int len)
{
    for (int i = 0; i < len; i++)
    {
        mat[i] = mat[i] > 0 ? mat[i] : 0;
    }
    return std::move(mat);
}

#endif