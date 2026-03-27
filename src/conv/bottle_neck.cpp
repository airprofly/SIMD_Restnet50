#include "bottle_neck.h"
#include "conv.h"
#include "bn.h"
#include "relu.h"

/**
 * @brief compute the add operation of the bottleneck layer(BTNK)

 * @param original the original feature  mat without convelution,batch norm and activation
 * @param conv_out the convelutioned feature mat
 * @param len the length of the feature mat

 * @return the bottleneck layer output feature mat
 * @note the conv_out mat will be freed
 */
aligned_unique_ptr<float> BTNK_add(aligned_unique_ptr<float> &original, // the original feature  mat without convelution,batch norm and activation
                                   aligned_unique_ptr<float> &conv_out, // the convelutioned feature mat
                                   int len                              // the length of the feature mat
)
{
    for (int i = 0; i < len; i++)
    {
        original[i] += conv_out[i];
    }
    conv_out.reset();

    return std::move(original); // return the original feature mat
}

/**
 * @brief compute the bottleneck layer(BTNK)

 * @param input the input feature mat
 * @param hei_in the height of the input feature mat
 * @param wid_in the width of the input feature mat
 * @param hei_out the height of the output feature mat
 * @param wid_out the width of the output feature mat
 * @param chan_out the channel of the output feature mat
 * @param layer_name the name of the bootleneck layer
 * @param is_down_sample the flag to indicate whether the input feature mat should be downsampled

 * @return the bottleneck layer output feature mat
 */
aligned_unique_ptr<float> bottleNeck_layer(aligned_unique_ptr<float> &input,         // the input feature mat
                                           int hei_in,                               // the height of the input feature mat
                                           int wid_in,                               // the width of the input feature mat
                                           int &hei_out,                             // the height of the output feature mat
                                           int &wid_out,                             // the width of the output feature mat
                                           int &chan_out,                            // the channel of the output feature mat
                                           const std::string &bottleNeck_layer_name, // the name of the bootleneck layer
                                           bool is_down_sample                       // the flag to indicate whether the input feature mat should be downsampled
)
{
    /**
     * first Conv + BN + RELU structure,the input will not be freed to skip
     *
     * this part will use kenerl size 1*1*channel_in_num to
     *      - reduce the channel number
     *      - linear combination of different channels
     */
    aligned_unique_ptr<float> conv1_out = compute_conv_layer(input, hei_in, wid_in, hei_out, wid_out, chan_out, bottleNeck_layer_name + "_conv1", false); // conv1
    conv1_out = bn_layer(conv1_out, hei_out, wid_out, chan_out, bottleNeck_layer_name + "_bn1");                                                          // bn1
    conv1_out = relu_layer<float>(conv1_out, hei_out * wid_out * chan_out);                                                                               // relu1

    /**
     *second Conv + BN + RELU structure,the input will be freed

     * this part will use kenerl size 3*3*channel_in_num to
     *    - expand the perceptual domain
     *    - linear combination of different channels
     */
    aligned_unique_ptr<float> conv2_out = compute_conv_layer(conv1_out, hei_out, wid_out, hei_out, wid_out, chan_out, bottleNeck_layer_name + "_conv2", true); // conv2
    conv2_out = bn_layer(conv2_out, hei_out, wid_out, chan_out, bottleNeck_layer_name + "_bn2");                                                               // bn2
    conv2_out = relu_layer<float>(conv2_out, hei_out * wid_out * chan_out);                                                                                    // relu2

    /**
     * third Conv + BN + RELU structure,the input will be freed
     *
     * this part will use kenerl size 1*1*channel_in_num to
     *      - increase the number of channels
     *      - linear combination of different channels
     */
    aligned_unique_ptr<float> conv3_out = compute_conv_layer(conv2_out, hei_out, wid_out, hei_out, wid_out, chan_out, bottleNeck_layer_name + "_conv3", true); // conv3
    conv3_out = bn_layer(conv3_out, hei_out, wid_out, chan_out, bottleNeck_layer_name + "_bn3");                                                               // bn3

    if (is_down_sample) // if the input feature need to be downsampled
    {
        aligned_unique_ptr<float> origin_conv_out = compute_conv_layer(input, hei_in, wid_in, hei_out, wid_out, chan_out, bottleNeck_layer_name + "_downsample_conv2d", true); // conv0
        origin_conv_out = bn_layer(origin_conv_out, hei_out, wid_out, chan_out, bottleNeck_layer_name + "_downsample_batchnorm");

        origin_conv_out = BTNK_add(origin_conv_out, conv3_out, hei_out * wid_out * chan_out);
        return std::move(relu_layer<float>(origin_conv_out, hei_out * wid_out * chan_out)); // relu3
    }else{
        input=BTNK_add(input, conv3_out, hei_out * wid_out * chan_out);
        return std::move(relu_layer<float>(input, hei_out * wid_out * chan_out));
    }
}
