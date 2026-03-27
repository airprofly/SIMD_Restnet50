#include<cmath>
#include<filesystem>

#include"bn.h"
#include"param.h"
#include"file_util.h"

/**
 * @brief compute the batch normalization layer

 * @param input the input feature mat
 * @param mean the mean of every channel
 * @param var the variance of every channel
 * @param gamma the scale parameter of every channel
 * @param bias the bias parameter of every channel
 * @param hei_in the height of the input feature mat
 * @param wid_in the width of the input feature mat
 * @param chan_in the channel of the input feature mat

 * @return the batch normalization result
 */
aligned_unique_ptr<float> bn(aligned_unique_ptr<float> &input, // the input feature mat
                                aligned_unique_ptr<float> &mean,  // the mean of every channel
                                aligned_unique_ptr<float> &var,   // the variance of every channel
                                aligned_unique_ptr<float> &gamma, // the scale parameter of every channel
                                aligned_unique_ptr<float> &bias,  // the bias parameter of every channel
                                int hei_in,                       // the height of the input feature mat
                                int wid_in,                       // the width of the input feature mat
                                int chan_in                       // the channel of the input feature mat
)
{
    for(int chan_idx{0}; chan_idx < chan_in; ++chan_idx)
    {
        float mean_cur{mean[chan_idx]};// the param of temp channel
        float var_cur{var[chan_idx]};
        float gamma_cur{gamma[chan_idx]};
        float bias_cur{bias[chan_idx]};

        for(int hei_idx{0}; hei_idx < hei_in; ++hei_idx){
            for(int wid_idx{0}; wid_idx < wid_in; ++wid_idx){
               int cur_idx{hei_idx * wid_in*chan_in+ wid_idx*chan_in + chan_idx};
               input[cur_idx] = (input[cur_idx] - mean_cur) / sqrt(var_cur +EPS);
               input[cur_idx]= input[cur_idx] * gamma_cur + bias_cur;
            }
        }

    }
    mean.reset();
    var.reset();
    gamma.reset();
    bias.reset();

    return std::move(input); // return the result
}



/**
 * @brief compute the batch normalization layer with para from file name provided

 * @param input the input feature mat
 * @param hei_in the height of the input feature mat
 * @param wid_in the width of the input feature mat
 * @param chan_in the channel of the input feature mat
 * @param layer_name the name of the layer

 * @return the batch normalization result
 */
aligned_unique_ptr<float> bn_layer(aligned_unique_ptr<float> &input, // the input feature mat
                                   int hei_in,                       // the height of the input feature mat
                                   int wid_in,                       // the width of the input feature mat
                                   int chan_in,                      // the channel of the input feature mat
                                   const std::string &layer_name     // the name of the layer
)
{
    const std::string weight_file_name="./model/resnet50_weight/resnet50_" + layer_name + "_weight.txt";
    const std::string bias_file_name="./model/resnet50_weight/resnet50_" + layer_name + "_bias.txt";
    const std::string mean_file_name="./model/resnet50_weight/resnet50_" + layer_name + "_running_mean.txt";
    const std::string var_file_name="./model/resnet50_weight/resnet50_" + layer_name + "_running_var.txt";

    namespace fs =std::filesystem;

    fs::path weight_path{weight_file_name};
    fs::path bias_path{bias_file_name};
    fs::path mean_path{mean_file_name};
    fs::path var_path{var_file_name};

    aligned_unique_ptr<float> gamma=load_data_from_file<float>(weight_path, chan_in);
    aligned_unique_ptr<float> bias=load_data_from_file<float>(bias_path, chan_in);
    aligned_unique_ptr<float> mean=load_data_from_file<float>(mean_path, chan_in);
    aligned_unique_ptr<float> var=load_data_from_file<float>(var_path, chan_in);

    return std::move(bn(input, mean, var, gamma, bias, hei_in, wid_in, chan_in));
}