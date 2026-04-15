# KuTACC

## 🔥Release Notes

- [2026/03] KuTACC项目首次上线，支持AlphaFold2应用底层计算融合算子、通信融合算子。

## 🚀概述

鲲鹏芯片支持向量、矩阵计算，带来算力提升的同时，辅以高速RDMA网络，带来超大带宽、微秒级延迟的极致性能。该芯片强浮点算力和高速带宽天然亲和AI推理计算。基于此，我们提出鲲鹏Transformer模型融合算子库（Kunpeng Unifined Transformer Accelerated Library，以下简称KuTACC），高效实现Transformer模型推理在鲲鹏处理器的执行。

## 📝版本配套

- 运行平台
    - 鲲鹏 920 专业版
- 系统规格
    - openEuler 22.03（LTS-SP4）AArch64

## ⚡️编译安装

若您希望**从零到一快速体验**项目能力，请参照下述编译安装教程。

#### 1. [获取 HPCKit 软件包](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/instg/KunpengHPCKit_install_007.html)
> https://www.hikunpeng.com/developer/hpc/hpckit-download

#### 2. [安装 HPCKit](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/instg/KunpengHPCKit_install_012.html)

##### 解压 HPCKit 软件包（HPCKit 版本号根据实际情况调整）
```
tar xvf HPCKit_26.0.RC1_Linux-aarch64.tar.gz
```
##### 安装 HPCKit
```
sh HPCKit_26.0.RC1_Linux-aarch64/install.sh -y --prefix=[HPCKit安装目录]
```

#### 3. [设置环境变量](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/instg/KunpengHPCKit_install_014.html)

##### 安装 Module 环境变量管理工具（需要已配置 yum 源）
```
yum install environment-modules
```

##### 加载 module
```
module use [HPCKit安装目录]/HPCKit/latest/modulefiles
```

##### 加载编译器环境变量（编译器版本号根据实际情况调整）
```
module load bisheng/compiler5.1.0.2/bishengmodule
```

##### 加载依赖库环境变量
```
module load bisheng/kutacc26.0.RC1/kutacc
module load bisheng/kupl26.0.RC1/release
module load bisheng/hmpi26.0.RC1/release
export CPLUS_INCLUDE_PATH=[HPCKit安装目录]/HPCKit/latest/hmpi/bisheng/release/hmpi/include:${CPLUS_INCLUDE_PATH}
```

#### 4. 安装编译所需依赖（需要已配置 yum 源）

安装 cmake
```
yum install cmake
```

#### 5. 源码编译安装

- 如果使用源码压缩包的代码下载方式
##### 上传并解压 KuTACC 项目源码包（分支名根据实际情况调整）
```
unzip kutacc-main.zip
```

##### 进入KuTACC项目源码根目录（分支名根据实际情况调整）
```
cd kutacc-main
```

##### 编译安装
可以使用build.sh将KuTACC安装在任意指定的路径下，支持Release或Debug模式的库安装。
- Release模式：
```
sh build.sh --install_path=/path/to/your/kutacc-path --build_type=Release
```
- Debug模式：
```
sh build.sh --install_path=/path/to/your/kutacc-path --build_type=Debug
```

## 📖学习教程

若您已学习**编译安装**，对本项目有一定认知，并希望**深入了解和体验项目**，请访问下述详细教程。

1. [开发指南](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/devg/KunpengHPCKit_developer_123.html)：提供详细接口开发指南，从零学习接口功能与开发。

## 🔍目录结构
项目详细目录介绍如下。
```
├── cmake                          # 项目工程编译脚本目录
├── include                        # 项目公共头文件
├── src                            # 项目源码目录
│   ├── activation                 # 激活函数
│   ├── attention                  # 计算融合算子
│   ├── comm                       # 通信融合算子
│   ├── linear                     # 线性算子
│   ├── math                       # 数学函数
│   ├── normalization              # 归一化函数
│   ├── softmax                    # softmax函数
│   ├── tensor                     # 张量数据结构
│   ├── utils                      # 通用功能工具
│   ├── version                    # 版本信息函数
│   ├── wrapper                    # 向量计算函数
│   └── CMakeLists.txt             # 源码编译配置文件
├── test                           # 测试工程目录
├── CMakeLists.txt
├── LICENSE
├── README.md
└── build.sh                       # 项目工程编译脚本
```

## 🤝联系我们

本项目功能和文档正在持续更新和完善中，建议您关注最新版本。

- **问题反馈**：通过 [【Issues】](https://atomgit.com/kunpengcompute/kutacc/issues)提交问题。
- **社区互动**：通过 [【鲲鹏论坛（HPC专区）】](https://www.hikunpeng.com/forum/forum-0187135482144798003-1.html)参与交流。
- **技术专栏**：通过 [【鲲鹏社区】](https://www.hikunpeng.com/developer/techArticles) 获取技术文章，如系列化教程、优秀实践等。