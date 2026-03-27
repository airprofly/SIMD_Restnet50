/**
File Name: file_util.h 
Author: airprofly
Date: 2025-04-01
Description: This file has realized the following functions:
            - get the file list of the test image
            - preprocess the image to adapt the resnet50 model
            - load the data from the file
*/

#ifndef _FILE_UTIL_H_
#define _FILE_UTIL_H_

#include <unordered_map>
#include <string>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "label.h"
#include "alignPtr.h"
#include"param.h"

/**
 * @brief get the file list of the path and the animal name for each file

 * @param path: the path of the file list

 * @return  the file list and the animal name for each file
 */
const std::unordered_map<std::filesystem::path, std::string> getFileLists(std::filesystem::path path);

/**
 * @brief preprocess the image to adapt the resnet50 model
 * - resize the image to 224*224
 * - normalize the image to [0,1]
 * - flatten the image to IMG_ROW_SIZE*IMG_COL_SIZE*3

 * @param img_path: the path of the image

 * @return the preprocessed image
 * @note the image is flattened to virtual size IMG_ROW_SIZE*IMG_COL_SIZE*3,it's actually flatted in line space
 */
aligned_unique_ptr<float> preProcessImg(const std::filesystem::path &img_path);

/**
 * @brief load the data from the file

 * @param path the path to the file
 * @param cnt the number of data to be loaded

 * @return the data stored in the mallocated aligned memory
 * @note the data will be freed automatically
 */
template <typename T>
aligned_unique_ptr<T> load_data_from_file(const std::filesystem::path &path, const int cnt)
{
    namespace fs = std::filesystem;
    if (cnt < 0)
    { // process the exception
        throw std::invalid_argument("cnt must be greater than 0");
        return nullptr;
    }
    if (!fs::exists(path))
    {
        throw std::runtime_error("file not exist\n" + path.string());
        return nullptr;
    }

    aligned_unique_ptr<T> data = make_aligned_unique<T>(cnt, ALIGN_SIZE); // malloc the aligned memory to store the data
    if (data == nullptr)
    { // if malloc failed,throw the exception
        throw std::runtime_error("failed to malloc memory");
        return nullptr;
    }

    /*open the file*/
    namespace fs = std::filesystem;
    std::ifstream fstream{path};
    if (!fstream.is_open())
    {
        throw std::runtime_error("failed to open file" + path.string());
        return nullptr;
    }
    float temp{0};//because the data in the file is float type,we must read it in float type first,otherwise the data will be wrong
    for(int i=0;i<cnt;i++){
        fstream >> temp;
        data[i] = static_cast<T>(temp); // read the data from the file
    }    

    fstream.close();
    return std::move(data);
}

/**
 * @brief show the image

 * @param img_path: the path of the image

 * @note the image will be shown and wait for the user to press any key to close the window
 */
void showIMG(const std::filesystem::path &img_path);

#endif