/**
File Name: param.h 
Author: airprofly
Date: 2025-04-01
Description: This file has realized the following functions:
            - defined the macros parameters for the model
*/


#ifndef _PARAM_H_
#define _PARAM_H_

/*the image size to be processed for the model*/
#define IMG_ROW_SIZE 224
#define IMG_COL_SIZE 224
#define IMG_SIZE (IMG_ROW_SIZE * IMG_COL_SIZE)

/* the param for the normalization the input image*/

/** the three channels' mean value of the trained model **/
#define IMG_RED_MEAN 0.485
#define IMG_GREEN_MEAN 0.456
#define IMG_BLUE_MEAN 0.406

/** the three channels' standard deviation of the trained model **/
#define IMG_RED_STD 0.229
#define IMG_GREEN_STD 0.224
#define IMG_BLUE_STD 0.225

/*the align size for SIMD*/
#define ALIGN_SIZE 32

/* the param size for the conv layer*/
#define CONV_PARAM_SIZE 5

#define EPS (1e-5)//avoid the zero division

#define OUTPUT_CLASS_NUM 1000 //the number of the classification

#define DISPLAY_CNT 10//the number of the feature map to be displayed

#define FLOAT_NUM 8 //the max number of the float data in the register 256

#endif