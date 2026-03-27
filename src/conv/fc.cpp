#include "fc.h"
#include "file_util.h"

#include <filesystem>

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
)
{
    const std::string fc_weight_file_name = "./model/resnet50_weight/resnet50_" + layer_name + "_weight.txt";
    const std::string fc_bias_file_name = "./model/resnet50_weight/resnet50_" + layer_name + "_bias.txt";

    namespace fs = std::filesystem;
    fs::path fc_weight_path{fc_weight_file_name};
    fs::path fc_bias_path{fc_bias_file_name};

    aligned_unique_ptr<float> weight = load_data_from_file<float>(fc_weight_path, len * OUTPUT_CLASS_NUM);
    aligned_unique_ptr<float> bias = load_data_from_file<float>(fc_bias_path, OUTPUT_CLASS_NUM);

    return std::move(fc(input, weight, bias, len));
}
