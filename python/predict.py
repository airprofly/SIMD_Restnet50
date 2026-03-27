"""
Description:
    This script uses PyTorch and a pre-trained ResNet50 model to classify images.

    It first loads the pre-trained ResNet50 model, then preprocesses all images in a specified directory and uses the model to make predictions.
    Finally, it outputs the top five most likely categories for each image.

Author:airprofly
Date:2025-04-13
"""
import torch  
import heapq  
import torchvision.models as models  

import os
model_path="D:\\torch_model\\resnet50.pth"
resnet50=models.resnet50(weights=None)
resnet50.load_state_dict(torch.load(model_path,weights_only=True))

resnet50.eval()  

import os  

# configure the directory of the images to be predicted
pic_dir = "pics/animals/" # the directory of the images to be predicted
# get all files in the directory
file_to_predict = [pic_dir + f for f in os.listdir(pic_dir) if os.path.isfile(pic_dir + f)]

from PIL import Image  # used for image processing
from torchvision import transforms  # used for image preprocessing

# iterate over all files in the directory to predict
for filename in file_to_predict:
    # open the image file
    input_image = Image.open(filename).convert("RGB")  # convert to RGB format
    print(input_image)
    # define the preprocessing steps
    preprocess = transforms.Compose(
        [
            transforms.Resize(224),  # just resize the image to 224x224
            # transforms.CenterCrop(224),  # center crop the image to 224x224
            transforms.ToTensor(),  # convert the image to a tensor
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # normalize the image
        ]
    )
    # preprocess the image
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model,because the model expects four-dimensional input

    # if there's a GPU available, move the input and model to GPU
    output = resnet50(input_batch)

    # get the top 5 predictions
    res = list(output[0].detach().numpy())
    index = heapq.nlargest(5, range(len(res)), res.__getitem__)

    print("predict picture: " + filename)
    # print the top 5 predictions labels
    with open("./python/imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
        for i in range(5):
            print("         top " + str(i + 1) + ": " + categories[index[i]])
            print(res[index[i]])
