<div align="center">

# SIMD ResNet50 图像分类
### SIMD-Optimized ResNet50 Image Classification Implementation

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/airprofly/SIMD_Restnet50) [![Star](https://img.shields.io/github/stars/airprofly/SIMD_Restnet50?style=social)](https://github.com/airprofly/SIMD_Restnet50/stargazers) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20) [![CMake](https://img.shields.io/badge/CMake-3.0+-brightgreen.svg)](https://cmake.org/) [![OpenCV](https://img.shields.io/badge/OpenCV-4.10-orange.svg)](https://opencv.org/)

[深度学习] · [SIMD优化] · [图像分类] · [ResNet50]

</div>

---

## 📖 项目简介

本项目是基于 ResNet50 的图像分类系统实现，使用现代 C++（C++20）从零构建了完整的神经网络推理框架。项目通过 SIMD（AVX2/FMA）指令集优化计算密集型操作，实现了高效的卷积神经网络前向推理。

**核心特性**：完全手写的神经网络层实现，包括卷积层、批归一化层、激活函数、池化层和全连接层，无需依赖深度学习框架即可完成图像分类任务。

## 📌 功能特性

- ✅ **完整 ResNet50 架构**：实现完整的 50 层残差网络，包含 16 个残差模块
- ✅ **SIMD 向量优化**：利用 AVX2/FMA 指令集加速矩阵运算和卷积操作
- ✅ **手写神经网络层**：卷积、批归一化、ReLU、池化、全连接层全部自主实现
- ✅ **模型权重导出**：支持从 PyTorch 预训练模型导出权重参数
- ✅ **多线程图像预处理**：并行处理图像加载和显示
- ✅ **跨平台支持**：使用 CMake 构建，支持 Windows/Linux 平台

## 📁 项目结构

<details>
<summary><b>查看完整目录结构</b></summary>

```text
SIMD_Restnet50/
├── CMakeLists.txt               # 🔧 CMake 构建配置
├── .clang-format                # 📐 代码格式化配置
├── .gitignore                   # 🚫 Git 忽略规则
├── README.md                    # 📄 项目说明文档
├── cmake/                       # 🔧 CMake 配置文件
│   └── ImgClassifyConfig.cmake.in
├── include/                     # 📂 头文件目录
│   ├── conv/                    # 🔗 卷积相关头文件
│   │   ├── bottle_neck.h        # 🍾 Bottleneck 残差模块
│   │   ├── bn.h                 # 📊 批归一化层
│   │   ├── conv.h               # 🔄 卷积层
│   │   ├── convAll.h            # 📦 卷积模块聚合头文件
│   │   ├── fc.h                 # 🔗 全连接层
│   │   ├── matrix.h             # 🧮 矩阵运算
│   │   ├── pool.h               # 🏊 池化层
│   │   └── relu.h               # ⚡ ReLU 激活函数
│   └── util/                    # 🛠️ 工具类头文件
│       ├── alignPtr.h           # 📍 内存对齐智能指针
│       ├── display.h            # 🖥️ 结果显示工具
│       ├── file_util.h          # 📄 文件处理工具
│       ├── label.h              # 🏷️ 标签定义
│       ├── param.h              # ⚙️ 参数配置
│       ├── print.h              # 🖨️ 打印工具
│       ├── timer.h              # ⏱️ 计时工具
│       └── utilAll.h            # 📦 工具模块聚合头文件
├── src/                         # 💻 源代码目录
│   ├── main.cpp                 # 🚀 主程序入口
│   ├── conv/                    # 🔗 卷积实现
│   │   ├── bottle_neck.cpp      # 🍾 Bottleneck 模块实现
│   │   ├── bn.cpp               # 📊 BN 层实现
│   │   ├── conv.cpp             # 🔄 卷积运算实现
│   │   ├── fc.cpp               # 🔗 FC 层实现
│   │   └── matrix.cpp           # 🧮 矩阵运算
│   └── util/                    # 🛠️ 工具实现
│       ├── alignPtr.cpp         # 📍 内存对齐工具
│       ├── display.cpp          # 🖥️ 显示功能
│       ├── file_until.cpp       # 📄 文件处理
│       └── print.cpp            # 🖨️ 打印功能
└── python/                      # 🐍 Python 辅助脚本
    ├── predict.py               # 🔮 PyTorch 预测脚本
    └── resnet50_parser.py       # 📝 模型权重导出脚本
```

</details>

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| **编程语言** | C++20 |
| **构建工具** | CMake 3.0+ |
| **图像处理** | OpenCV 4.10 |
| **SIMD 指令集** | AVX2, FMA |
| **Python 工具** | PyTorch (用于模型训练与权重导出) |

## 🔧 环境配置

### 前置要求

| 依赖项 | 最低版本 | 推荐版本 |
|--------|----------|----------|
| C++ 编译器 | 支持 C++20 | GCC 10+, Clang 12+, MSVC 2019+ |
| CMake | 3.0 | 3.20+ |
| OpenCV | 4.0 | 4.10 |

### Windows 平台

1. **安装 MinGW 或 MSVC 编译器**
   ```bash
   # 推荐使用 MinGW-w64
   # 或安装 Visual Studio 2019/2022
   ```

2. **编译 OpenCV**
   ```bash
   # 使用 MinGW 编译 OpenCV（推荐）
   # 或下载预编译的 OpenCV 库
   ```

3. **修改 CMakeLists.txt 中的 OpenCV 路径**
   ```cmake
   set(OPENCV_PATH "你的OpenCV路径/install/x64/mingw/lib")
   ```

### Linux 平台

```bash
# 安装依赖
sudo apt-get update
sudo apt-get install -y build-essential cmake libopencv-dev

# 验证安装
g++ --version
cmake --version
pkg-config --modversion opencv4
```

## 🚀 快速开始

### 方式一：从源码编译

```bash
# 1. 克隆仓库
git clone https://github.com/airprofly/SIMD_Restnet50.git
cd SIMD_Restnet50

# 2. 创建构建目录
mkdir build && cd build

# 3. 配置项目（根据平台选择）
# Windows (MinGW):
cmake -G "MinGW Makefiles" -DOPENCV_PATH="你的OpenCV路径" ..

# Windows (MSVC):
cmake -G "Visual Studio 16 2019" -A x64 ..

# Linux:
cmake -DCMAKE_BUILD_TYPE=Release ..

# 4. 编译项目
cmake --build . --config Release

# 5. 运行程序
cd bin
./ImgClassify
```

### 方式二：VSCode 任务

如果你使用 VSCode，可以直接按 `Ctrl+Shift+B` 编译项目。

## 📊 使用说明

### 修改测试图片路径

在 [src/main.cpp](src/main.cpp) 中修改图片路径：

```cpp
const std::string path_string = "pics/animals"; // 修改为你的图片目录
```

### 程序输出

程序会依次输出每一层的特征图尺寸和预测结果：

```
the current predict animal is lion
the conv1 layer output size is 112 112 64
the layer1 output size is 56 56 256
the layer2 output size is 28 28 512
the layer3 output size is 14 14 1024
the layer4 output size is 7 7 2048
the global avgpool scuessfully
the fc layer scuessfully
total time: XXX ms
```

## 🔬 网络架构

<details>
<summary><b>查看 ResNet50 架构详情</b></summary>

### 整体结构

```
输入图像 (224×224×3)
    ↓
卷积层 (7×7, 64, stride=2) + BN + ReLU + MaxPool
    ↓
Layer1: 3个Bottleneck块 (输出56×56×256)
    ↓
Layer2: 4个Bottleneck块 (输出28×28×512)
    ↓
Layer3: 6个Bottleneck块 (输出14×14×1024)
    ↓
Layer4: 3个Bottleneck块 (输出7×7×2048)
    ↓
全局平均池化 (Global Average Pooling)
    ↓
全连接层 (FC) → Softmax
    ↓
输出分类结果 (1000类)
```

### Bottleneck 残差模块

每个 Bottleneck 包含三层卷积：
1. 1×1 卷积（降维）
2. 3×3 卷积（特征提取）
3. 1×1 卷积（升维）

残差连接将输入直接加到输出，缓解梯度消失问题。

</details>

## ⚡ SIMD 优化

本项目使用以下 SIMD 指令集加速计算：

| 指令集 | 功能 | 性能提升 |
|--------|------|----------|
| AVX2 | 256位向量同时处理8个float | ~4x |
| FMA | 融合乘加运算 | ~2x |

启用 SIMD 后，卷积运算和矩阵乘法性能显著提升。在 [CMakeLists.txt](CMakeLists.txt) 中默认启用 SIMD 优化：

```cmake
option(ENABLE_SIMD "enable SIMD" ON)
```

## 🐍 Python 工具

### 模型权重导出

将 PyTorch 预训练模型权重导出为文本格式：

```bash
cd python
python resnet50_parser.py
```

### PyTorch 预测对比

使用 PyTorch 进行预测对比验证：

```bash
cd python
python predict.py
```

## 📈 性能

| 配置 | 推理时间 |
|------|----------|
| 无SIMD优化 | ~8000 ms |
| AVX2+FMA优化 | ~2000 ms |

*测试环境：Intel Core i7-10750H, 单线程推理*

## 📄 许可证

本项目采用 [MIT 许可证](https://opensource.org/licenses/MIT)。

## 🙏 致谢

- [ResNet](https://arxiv.org/abs/1512.03385) 论文作者
- [OpenCV](https://opencv.org/) 开源社区
- [PyTorch](https://pytorch.org/) 深度学习框架

## 📚 参考资料

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). CVPR.
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [ImageNet 数据集](https://www.image-net.org/)
