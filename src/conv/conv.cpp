#include "conv.h"
#include "matrix.h"
#include"print.h"

/**
 * @brief compute the convolution layer

 * @param input the input feature mat
 * @param hei_in the height of the input feature mat
 * @param wid_in the width of the input feature mat
 * @param hei_out the height of the output feature mat
 * @param wid_out the width of the output feature mat
 * @param chan_out the channel of the output feature mat
 * @param layer_name the name of the layer
 * @param is_free_input the flag to indicate whether the input feature mat should be freed

 * @return the output feature mat pointer
 * @note the output mat is stored in format [hei_out, wid_out, chan_out],just HWC format
 */
aligned_unique_ptr<float> compute_conv_layer(aligned_unique_ptr<float>& input, // the input feature mat
                                             int hei_in,                      // the height of the input feature mat
                                             int wid_in,                      // the width of the input feature mat
                                             int &hei_out,                    // the height of the output feature mat
                                             int &wid_out,                    // the width of the output feature mat
                                             int &chan_out,                   // the channel of the output feature mat
                                             const std::string& layer_name,   // the name of the layer
                                             bool is_free_input               // the flag to indicate whether the input feature mat should be freed
)
{
    namespace fs = std::filesystem;
    // std::string conv_param = "../../model/resnet50_weight/resnet50_" + layer_name + "_param.txt"; // the path of the conv param,store the conv like [ci,co,kernel,stride,pad]
    std::string conv_param = "./model/resnet50_weight/resnet50_" + layer_name + "_param.txt"; // the path of the conv param,store the conv like [ci,co,kernel,stride,pad]
    fs::path conv_param_path{conv_param};

    /*load the conv param,like channel_in,channel_out,kernel,stride,pad*/
    aligned_unique_ptr<int> param = load_conv_param(conv_param_path, CONV_PARAM_SIZE);
    // print_conv_param(param, CONV_PARAM_SIZE, conv_param);

    int chan_in = param[0];  // the input channel
    chan_out = param[1]; // the output channel
    int kernel = param[2];      // the kernel size
    int stride = param[3];      // the stride
    int pad = param[4];         // the padding

    /*load the conv weight*/
    std::string conv_weight = "./model/resnet50_weight/resnet50_" + layer_name + "_weight.txt"; // the path of the conv weight
    fs::path conv_weight_path{conv_weight};
    aligned_unique_ptr<float> weight = load_conv_weight(conv_weight_path, chan_in * kernel * kernel * chan_out);
    
    // print_conv_weight(weight, channel_in, channel_out, kernel, conv_weight); // print the conv weight


    if (hei_in == IMG_ROW_SIZE && (wid_in == IMG_COL_SIZE))
    {
        return std::move(conv2d(input,weight, hei_in, wid_in, chan_in, hei_out, wid_out, chan_out, kernel, stride, pad, true, is_free_input));
    }
    else
    {
        return std::move(conv2d(input,weight, hei_in, wid_in, chan_in, hei_out, wid_out, chan_out, kernel, stride, pad, false, is_free_input));
    }
}