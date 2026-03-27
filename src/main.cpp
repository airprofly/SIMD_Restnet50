/**
File Name: main.cpp
Author: airprofly
Email: 3502196756@qq.com
Date: 2025-04-01
Description:This file has realized the following functions:
            - using the resnet50 model to predict the animal
*/

#include <iostream>
#include<immintrin.h>
#include<opencv2/opencv.hpp>
#include<thread>

#include "convAll.h"
#include "utilAll.h"

int main(int argc, char const *argv[])
{
    /*get the test file path and the corresponding animal names*/
    const std::string path_string = "E:/codeFiles/img_divide/pics/animals"; // the test file path
    std::filesystem::path path{path_string};

    std::unordered_map<std::filesystem::path, std::string> file_lists; // the map to store the file path and the corresponding animal names
    file_lists = getFileLists(path);

    for (auto file : file_lists)
    {
        std::cout << "the current predict animal is " << file.second << std::endl; // print the current predict animal info

        /* preprocess the image to adapt the resnet50 model */

        aligned_unique_ptr<float> img_data = preProcessImg(file.first);
        std::thread t_showImg(showIMG, file.first); // show the image
        t_showImg.detach(); // detach the thread


        int height_in, width_in, channel_in;    // store the input feature map size
        int height_out, width_out, channel_out; // the output feature map size

        auto start_time = getTime(); // get the start time

        img_data = compute_conv_layer(img_data, IMG_ROW_SIZE, IMG_COL_SIZE, height_out, width_out, channel_out, "conv1"); // convelution free the input data
        std::cout << "the conv1 layer output size is " << height_out << " " << width_out << " " << channel_out << std::endl;
        // the output feature map format[height_out, width_out, channel_out] is 112*112*64
        img_data = bn_layer(img_data, height_out, width_out, channel_out, "bn1");
        img_data = maxPool_layer<float>(img_data, channel_out, height_out, width_out, height_out, width_out, 3, 2, 1); // maxpool layer with the kernel size 3*3, stride 2, padding 1
        // the  output feature map format[height_out, width_out, channel_out] is 56*56*64

        /**
         * layer 1
         *
         * contains 3 bottleneck layers
         *
         * - the fisrt bottleneck layer:
         *    - downsample the input feature map, stride=1 because the feature map need to resize
         *    -the ouput feature map size is 56*56*64(with 64 kenerl 1*1*64)->56*56*64(with 64 kenerl 3*3*64)->56*56*256(with 256 kenerl 1*1*64)
         * - the second bottleneck layer:
         *    - no downsample, stride =1
         *    -the ouput feature map size is 56*56*64(with 64 kenerl 1*1*256)->56*56*64(with 64 kenerl 3*3*64)->56*56*256(with 256 kenerl 1*1*64)
         * - the third bottleneck layer:
         *    - no downsample, stride =1
         *    - the ouput feature map size is 56*56*64(with 64 kenerl 1*1*256)->56*56*64(with 64 kenerl 3*3*64)->56*56*256(with 256 kenerl 1*1*64)
         */
        img_data = bottleNeck_layer(img_data, height_out, width_out, height_out, width_out, channel_out, "layer1_bottleneck0", true);  // the bottleneck layer 1 need to downsample the input feature map
        img_data = bottleNeck_layer(img_data, height_out, width_out, height_out, width_out, channel_out, "layer1_bottleneck1", false); // the bottleneck layer 2 no need to downsample the input feature map
        img_data = bottleNeck_layer(img_data, height_out, width_out, height_out, width_out, channel_out, "layer1_bottleneck2", false); // the bottleneck layer 3 no need to downsample the input feature map
        std::cout << "the layer1 output size is " << height_out << " " << width_out << " " << channel_out << std::endl;
        /**
         * layer 2
         *
         * contains 4 bottleneck layers
         *
         * - the fisrt bottleneck layer:change to 28*28*256
         *
         */
        img_data = bottleNeck_layer(img_data, height_out, width_out, height_out, width_out, channel_out, "layer2_bottleneck0", true);  // the bottleneck layer 1 need to downsample the input feature map
        img_data = bottleNeck_layer(img_data, height_out, width_out, height_out, width_out, channel_out, "layer2_bottleneck1", false); // the bottleneck layer 2 no need to downsample the input feature map
        img_data = bottleNeck_layer(img_data, height_out, width_out, height_out, width_out, channel_out, "layer2_bottleneck2", false); // the bottleneck layer 3 no need to downsample the input feature map
        img_data = bottleNeck_layer(img_data, height_out, width_out, height_out, width_out, channel_out, "layer2_bottleneck3", false);
        std::cout << "the layer2 output size is " << height_out << " " << width_out << " " << channel_out << std::endl;
        
        /**
         * layer 3
         *
         * contains 6 bottleneck layers
         *
         * - the fisrt bottleneck layer:change to 14*14*1024
         */
        img_data = bottleNeck_layer(img_data, height_out, width_out, height_out, width_out, channel_out, "layer3_bottleneck0", true);  // the bottleneck layer 1 need to downsample the input feature map
        img_data = bottleNeck_layer(img_data, height_out, width_out, height_out, width_out, channel_out, "layer3_bottleneck1", false); // the bottleneck layer 2 no need
        img_data = bottleNeck_layer(img_data, height_out, width_out, height_out, width_out, channel_out, "layer3_bottleneck2", false);
        img_data = bottleNeck_layer(img_data, height_out, width_out, height_out, width_out, channel_out, "layer3_bottleneck3", false);
        img_data = bottleNeck_layer(img_data, height_out, width_out, height_out, width_out, channel_out, "layer3_bottleneck4", false);
        img_data = bottleNeck_layer(img_data, height_out, width_out, height_out, width_out, channel_out, "layer3_bottleneck5", false);
        std::cout<< "the layer3 output size is " << height_out << " " << width_out << " " << channel_out << std::endl;

        /**
         * layer 4
         *
         * contains 3 bottleneck layers
         *
         * - the fisrt bottleneck layer:change to 7*7*2048
         */
        img_data = bottleNeck_layer(img_data, height_out, width_out, height_out, width_out, channel_out, "layer4_bottleneck0", true);  // the bottleneck layer 1 need to downsample the input feature map
        img_data = bottleNeck_layer(img_data, height_out, width_out, height_out, width_out, channel_out, "layer4_bottleneck1", false); // the bottleneck layer 2 no need
        img_data = bottleNeck_layer(img_data, height_out, width_out, height_out, width_out, channel_out, "layer4_bottleneck2", false); // the bottleneck layer 3 no need
        std::cout<< "the layer4 output size is " << height_out << " " << width_out << " " << channel_out << std::endl;
        

        /**
         *global avgpool layer (GAP)
         *
         * the output feature map size is 1*1*2048,and in this process has been flattened to 2048*1*1
         */
        img_data = avgPool_layer<float>(img_data, channel_out, height_out, width_out, height_out, width_out, 7, 1, 0); // avgpool layer with the kernel size 7*7, stride 1, padding 0
        std::cout<<"the global avgpool scuessfully"<<std::endl;

        /* fc layer using general matrix multiplication(GEMM) */
        img_data = fc_layer(img_data, "fc", 2048);
        std::cout<<"the fc layer scuessfully"<<std::endl;

        int end = getTime();               // get the end time
        int total_time = end - start_time; // get the time cos
        std::cout << "total time: " << total_time << " ms" << std::endl
                  << std::endl;

        /* show the result */
        display(img_data, OUTPUT_CLASS_NUM);
        
        break;
    }

    cv::waitKey(0);
    return 0;
}
