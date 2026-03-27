"""
Description:
    This file is used to parse the ResNet50 model and store the weights and parameters of each layer in text files.

Author:airprofly
Date:2023/10/12
"""

import numpy as np
from torchvision import models
import torch


model_path = "./model/resnet50.pth"
resnet50 = models.resnet50(weights=None)
resnet50.load_state_dict(torch.load(model_path, weights_only=True))

resnet50.eval()
print(resnet50)


# define the directory to save the weights
dump_dir = "./model/resnet50_weight/"


def save_conv_param(data, file):
    """
    # Description:
        This function saves the parameters of a convolutional layer to a text file.
    Args:
        data (torch.nn.Conv2d): The convolutional layer object from PyTorch.
        file (str): The base name of the file to save the parameters.
    """
    # get the kernel size, stride, padding, input channels, and output channels
    kh = data.kernel_size[0]  # kernel height
    sh = data.stride[0]  # stride
    pad_l = data.padding[0]  # padding
    ci = data.in_channels  # channel input size
    co = data.out_channels  # channel output size

    # combine these parameters into a list
    l = [ci, co, kh, sh, pad_l]

    # save these parameters to a text file using numpy
    np.savetxt(dump_dir + file + str("_param.txt"), l)


def save_bn_param(data, file):
    """
    # Description:
        This function saves the parameters of a batch normalization layer to a text file.
    Args:
        data (torch.nn.BatchNorm2d): The batch normalization layer object from PyTorch.
        file (str): The base name of the file to save the parameters.
    """
    # get the number of features, epsilon, and momentum
    eps = data.eps  # epsilon，used for numerical stability
    momentum = data.momentum  # momentum, used for running mean and variance

    # combine these parameters into a list
    l = [eps, momentum]

    # get the number of features from the weight tensor
    # the weight tensor is a 1D tensor with size equal to the number of features
    np.savetxt(dump_dir + file + "_param.txt", l)


def save(data, file):
    """
    # Description:
         This function saves the weights and parameters of a layer to text files.
    Args:
         data (torch.nn.Module): The layer object from PyTorch.
         file (str): The base name of the file to save the parameters.
    """
    # if the layer is a convolutional layer
    if isinstance(data, type(resnet50.conv1)):
        # store the parameters of the convolutional layer
        save_conv_param(data, file)
        # save the weights of the convolutional layer
        # convert the weights to a numpy array and transpose it
        w = np.array(data.weight.data.cpu().numpy())
        w = np.transpose(w, (0, 2, 3, 1))
        # save the weights as a flattened array
        np.savetxt(dump_dir + file + "_weight.txt", w.reshape(-1, 1))

    # if the layer is a batch normalization layer
    if isinstance(data, type(resnet50.bn1)):
        # store the parameters of the batch normalization layer
        save_bn_param(data, file)
        # save the running mean and variance
        m = np.array(data.running_mean.data.cpu().numpy())
        np.savetxt(dump_dir + file + "_running_mean.txt", m.reshape(-1, 1))

        v = np.array(data.running_var.data.cpu().numpy())
        np.savetxt(dump_dir + file + "_running_var.txt", v.reshape(-1, 1))

        # save the weights of the batch normalization layer
        b = np.array(data.bias.data.cpu().numpy())
        np.savetxt(dump_dir + file + "_bias.txt", b.reshape(-1, 1))

        w = np.array(data.weight.data.cpu().numpy())
        np.savetxt(dump_dir + file + "_weight.txt", w.reshape(-1, 1))

    # if the layer is a fully connected layer
    if isinstance(data, type(resnet50.fc)):
        # store the parameters of the fully connected layer
        print(data.weight.shape)
        # save the weights of the fully connected layer
        bias = np.array(data.bias.data.cpu().numpy())
        np.savetxt(dump_dir + file + "_bias.txt", bias.reshape(-1, 1))

        w = np.array(data.weight.data.cpu().numpy())
        np.savetxt(dump_dir + file + "_weight.txt", w.reshape(-1, 1))


# this function traverses each residual block in the ResNet50 model and saves the weights and parameters of the convolutional layers and batch normalization layers using the save function.
# it also saves the weights and parameters of the downsample layers if they exist.
# the function takes two arguments: layer, which is a residual block in the ResNet50 model, and layer_index, which is the index of the residual block used for file naming.
def save_bottle_neck(layer, layer_index):
    """
    # Description:
        This function traverses each residual block in the ResNet50 model and saves the weights and parameters of the convolutional layers and batch normalization layers using the save function.
        It also saves the weights and parameters of the downsample layers if they exist.

    Args:
        layer (torch.nn.Module): The residual block object from PyTorch.
        layer_index (int): The index of the residual block used for file naming.
    """
    bottle_neck_idx = 0  # initialize the index for the residual block

    # create a layer name for the residual block
    layer_name = "resnet50_layer" + str(layer_index) + "_bottleneck"

    # iterate through each residual block in the layer
    for bottleNeck in layer:
        # save the weights and parameters of the convolutional layers and batch normalization layers
        save(bottleNeck.conv1, layer_name + str(bottle_neck_idx) + "_conv1")
        save(bottleNeck.bn1, layer_name + str(bottle_neck_idx) + "_bn1")
        save(bottleNeck.conv2, layer_name + str(bottle_neck_idx) + "_conv2")
        save(bottleNeck.bn2, layer_name + str(bottle_neck_idx) + "_bn2")
        save(bottleNeck.conv3, layer_name + str(bottle_neck_idx) + "_conv3")
        save(bottleNeck.bn3, layer_name + str(bottle_neck_idx) + "_bn3")

        # if the residual block has a downsample layer, save its weights and parameters
        if bottleNeck.downsample:
            save(
                bottleNeck.downsample[0],
                layer_name + str(bottle_neck_idx) + "_downsample_conv2d",
            )
            save(
                bottleNeck.downsample[1],
                layer_name + str(bottle_neck_idx) + "_downsample_batchnorm",
            )

        # increment the index for the next residual block
        bottle_neck_idx += 1


# save the weights and parameters of the first convolutional layer and batch normalization layer
save(resnet50.conv1, "resnet50_conv1")
# save the weights and parameters of the first batch normalization layer
save(resnet50.bn1, "resnet50_bn1")

# save the weights and parameters of the first max pooling layer
save_bottle_neck(resnet50.layer1, 1)  # save the first bottle neck
save_bottle_neck(resnet50.layer2, 2)  # save the second bottle neck
save_bottle_neck(resnet50.layer3, 3)  # save the third bottle neck
save_bottle_neck(resnet50.layer4, 4)  # save the fourth bottle neck

# store the weights and parameters of the average pooling layer
save(resnet50.fc, "resnet50_fc")

print("save successfully")