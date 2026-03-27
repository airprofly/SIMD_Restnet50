# Image Classification
This repository contains the code for image classification using C and C++  constructing resnet50 model.
## Table of Contents
<div align="center">

| DIRECTORY | FUNCTION |
| :---: | :---: |
| include | the header files |
| src | the source files |
|pics | the pictures for testing the model|
|model|the model parameters and the structure|
|python|the python code for training the model and store the model|
</div>

## Attention
There are two points need to be paid attention to before compiling:
1.The OpenCV library is required to be compiled by MinGW(If you are using the tasks provided in this project).Or you could modify compiler in the tasks.json to MSVC. 

2.And you should modify the path of the OpenCV library in the CMakeLists.txt file.


## Run the project
When you want to run the project, you should confirm that you have completed the attention mentioned above.Then you could use "Ctrl+Shift+B" to compile the project.And find the executable file in the build/bin/ directory.