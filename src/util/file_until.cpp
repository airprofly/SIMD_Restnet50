#include <iostream>
#include <opencv2/opencv.hpp>

#include "file_util.h"
#include "param.h"
#include "alignPtr.h"

/**
 * @brief get the file list of the path and the animal name for each file

 * @param path: the path of the file list

 * @return  the file list and the animal name for each file
 */
const std::unordered_map<std::filesystem::path, std::string> getFileLists(std::filesystem::path path)
{
    namespace fs = std::filesystem;

    if (!fs::exists(path)) // if the path is not exist,cerr the error message
    {
        std::cerr << path << std::endl;
        std::cerr << "The path is not exist!" << std::endl;
        return {};
    }

    std::unordered_map<fs::path, std::string> fileMap; // the file path and the corresponding animal name map
    int label = 0;

    for (const auto &entry : fs::directory_iterator(path))
    {
        if (!entry.is_regular_file())
            continue;
        std::string baseName = entry.path().stem().string(); // get the file base name
        fileMap.emplace(entry.path(), baseName);             // insert the file path and the corresponding animal name into the map
    }
    return fileMap;
}

/**
 * @brief preprocess the image to adapt the resnet50 model
 * @brief - resize the image to 224*224
 * @brief - normalize the image to [0,1]
 * @brief - flatten the image to IMG_ROW_SIZE*IMG_COL_SIZE*3

 * @param img_path: the path of the image

 * @return the preprocessed image
 * @note the image is flattened to virtual size IMG_ROW_SIZE*IMG_COL_SIZE*3,it's actually flatted in line space
 */
aligned_unique_ptr<float> preProcessImg(const std::filesystem::path &img_path)
{
    if (!std::filesystem::exists(img_path))
    { // judge if the file path is exist
        std::cerr << img_path << std::endl;
        std::cerr << "The file is not exist!" << std::endl;
        return nullptr;
    }

    /*read the image*/
    cv::Mat img = cv::imread(img_path.string(), cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cerr << "Failed to read image!" << std::endl;
        std::cerr << img_path << std::endl;
        return nullptr;
    }

    /*resize the image to 224*224*/
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);                  // convert the image from BGR to RGB
    cv::resize(img, img, cv::Size(IMG_ROW_SIZE, IMG_COL_SIZE)); // resize the image to IMG_ROW_SIZE*IMG_COL_SIZE

    /*normalize the image and flatten the image*/
    aligned_unique_ptr<float> img_ret = make_aligned_unique<float>(IMG_ROW_SIZE * IMG_COL_SIZE * 3, ALIGN_SIZE); // allocate the aligned memory to store the flattened image

    // MARK:the image data is in RGB format,and the data is three channels,so I could't come up with processing method with SIMD
    for (int i = 0; i < IMG_ROW_SIZE; ++i)
    {
        for (int j = 0; j < IMG_COL_SIZE; ++j)
        {
            img_ret[i * IMG_COL_SIZE * 3 + j * 3 + 0] = (img.at<cv::Vec3b>(i, j)[0] / 255.0f - IMG_RED_MEAN) / IMG_RED_STD;     // R
            img_ret[i * IMG_COL_SIZE * 3 + j * 3 + 1] = (img.at<cv::Vec3b>(i, j)[1] / 255.0f - IMG_GREEN_MEAN) / IMG_GREEN_STD; // G
            img_ret[i * IMG_COL_SIZE * 3 + j * 3 + 2] = (img.at<cv::Vec3b>(i, j)[2] / 255.0f - IMG_BLUE_MEAN) / IMG_BLUE_STD;   // B
        }
    }
    return std::move(img_ret); // return the preprocessed image
}

/**
 * @brief show the image

 * @param img_path: the path of the image

 * @note the image will be shown and wait for the user to press any key to close the window
 */
void showIMG(const std::filesystem::path &img_path){
    std::string img_name = img_path.stem().string(); // get the file base name
    cv::Mat img = cv::imread(img_path.string(), cv::IMREAD_UNCHANGED); // read the image
    if(img.empty())
    {
        std::cout << "the image is empty" << std::endl;
        return;
    }else{
        cv::imshow(img_name,img); // show the image
        cv::waitKey(0); // wait for the user to press any key to close the window
    }
}
