/**
File Name: display.h
Author: airprofly
Email: 3502196756@qq.com
Date: 2025-04-02
Description: This file has realized the following functions:
              - sort the labels according to the probability
*/

#ifndef _DISPLAY_H_
#define _DISPLAY_H_

#include "alignPtr.h"

/**
 * @brief display the top 5 labels according to the probability

 * @param input the input array of the probability
 * @param cnt the count of the possible labels

 */
void display(aligned_unique_ptr<float> &input, int cnt);

#endif