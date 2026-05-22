# KuTACC

## 🔥Release Notes

- [2026/03] The Kunpeng Unified Transformer Accelerated Library (KuTACC) project went live, introducing support for AlphaFold2 through underlying compute and communication fused operator optimizations.

## 🚀Overview

With support for vector and matrix computations, Kunpeng chips offer enhanced computing power alongside high-speed RDMA networks for ultra-wide bandwidth and microsecond latency. These strengths—robust floating-point performance and high bandwidth—are inherently optimized for AI inference. Based on this, we developed KuTACC to efficiently execute Transformer model inference on the Kunpeng platform.

## 📝Version Mapping

- Operating platform
    - Kunpeng 920 Pro
- System specifications
    - openEuler 22.03 LTS SP4 AArch64

## ⚡️Build and Installation

Follow the build and installation guide below for a quick, start-from-scratch experience of the project.

### 1. [Obtain the HPCKit software package](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/instg/KunpengHPCKit_install_007.html).

> https://www.hikunpeng.com/developer/hpc/hpckit-download

### 2. [Install HPCKit](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/instg/KunpengHPCKit_install_012.html).

#### Extract the HPCKit software package (replace the version number in the example with your actual version).

```
tar xvf HPCKit_26.0.RC1_Linux-aarch64.tar.gz
```

#### Install HPCKit.

```
sh HPCKit_26.0.RC1_Linux-aarch64/install.sh -y --prefix=[HPCKit_installation_directory]
```

### 3. [Set environment variables](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/instg/KunpengHPCKit_install_014.html).

##### Install the Module environment management tool (requires yum repository to be configured).
```
yum install environment-modules
```

#### Load the module.

```
module use [HPCKit_installation_directory]/HPCKit/latest/modulefiles
```

#### Load the environment variables of the compiler (replace the version number in the example with your actual version).

```
module load bisheng/compiler5.1.0.2/bishengmodule
```

#### Load the environment variables of dependencies.

```
module load bisheng/kutacc26.0.RC1/kutacc
module load bisheng/kupl26.0.RC1/release
module load bisheng/hmpi26.0.RC1/release
export CPLUS_INCLUDE_PATH=[HPCKit_installation_directory]/HPCKit/latest/hmpi/bisheng/release/hmpi/include:${CPLUS_INCLUDE_PATH}
```

### 4. Install the dependencies required for build (requires yum repository to be configured).

Install CMake.

```
yum install cmake
```

### 5. Build and install from source.

- If you choose to download the source code package
##### Upload and extract the KuTACC project source package (replace the branch name in the example with your actual branch).
```
unzip kutacc-main.zip
```

##### Navigate to the root directory of the KuTACC project source code (replace the branch name in the example with your actual branch).
```
cd kutacc-main
```

##### Build and install.
Go to the KuTACC root directory and use `build.sh` to install KuTACC to any specified path. Both release and debug modes are supported.
- Release mode：
```
sh build.sh --install_path=/path/to/your/kutacc-path --build_type=Release
```
- Debug mode：
```
sh build.sh --install_path=/path/to/your/kutacc-path --build_type=Debug
```

## 📖Tutorials

If you are familiar with the **build and installation process** and would like to **gain a deeper understanding of the project**, please visit the following detailed tutorials:

[Developer Guide](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/devg/KunpengHPCKit_developer_123.html): Master API functionality and development with in-depth instructions tailored for developers of all levels.

## 🔍Directory Structure

The project directory structure is as follows:

```
├── cmake                          # Project build script directory
├── include                        # Project common header files
├── src                            # Project source directory
│   ├── activation                 # Activation function
│   ├── attention                  # Compute fused operator
│   ├── comm                       # Communication fused operator
│   ├── linear                     # Linear operator
│   ├── math                       # Math functions
│   ├── normalization              # Normalization function
│   ├── softmax                    # Softmax function
│   ├── tensor                     # Tensor data structure
│   ├── utils                      # Common utilities
│   ├── version                    # Version information function
│   ├── wrapper                    # Vector computation function
│   └── CMakeLists.txt             # Source build configuration file
├── test                           # Project test directory
├── CMakeLists.txt
├── LICENSE
├── README.md
└── build.sh                       # Project build script
```

## 🤝Contact Us

Features and documentation are updated regularly. Please follow the latest version for the most up-to-date information.

- **Issue feedback**: Submit queries or report bugs via [Issues](https://atomgit.com/kunpengcompute/kutacc/issues).
- **Community interaction**: Join discussions and share ideas via [Discussions](https://atomgit.com/kunpengcompute/kutacc/discussions).
- **Technical columns**: Access in-depth technical articles, serialized tutorials, and best practices through the [Kunpeng Community](https://www.hikunpeng.com/developer/techArticles).
