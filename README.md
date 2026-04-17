# Kunpeng Unifined Transformer Accelerated Library

## 1.简介
鲲鹏芯片支持向量、矩阵计算，带来算力提升的同时，辅以高速RDMA网络，带来超大带宽、微秒级延迟的极致性能。该芯片强浮点算力和高速带宽天然亲和
AI推理计算。基于此，我们提出一种鲲鹏平台上Transformer模型融合算子库（简称“KuTACC”），高效实现Transformer模型推理在鲲鹏处理器的执行。

## 2.本地运行

### 2.1 依赖软件安装
该方案是用HPCKit组件中的毕昇编译器进行编译，HPCKit安装流程参考[官方指导文档](https://www.hikunpeng.com/developer/hpc/hpckit-download)。

KuTACC的安装需要使用HPCKit环境中的毕昇编译器、KUPL，配置流程参考[HPCKit介绍](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/devg/KunpengHPCKit_developer_002.html)。

### 2.2 源码编译与安装
执行下述命令进行源码编译与安装，默认安装在`/path_to_kutacc/install`目录下。
```shell
sh build.sh
```

### 2.3 环境变量配置
执行下述命令设置KuTACC的环境变量，而后在目标应用中编译使用即可。
```shell
export KUTACC_LIB=/path_to_kutacc/install/lib
export KUTACC_INCLUDE=/path_to_kutacc/install/include
```

### 2.4 运行测试用例（可选）
执行下述命令运行KuTACC的测试用例，验证目标环境是否满足KuTACC的要求。
```shell
cd test
bash build.sh
bash run_test.sh
```

## 3. 支持的应用

| 支持的应用    | 应用版本 |
|:---------|-----:|
| DeepSeek |   V3/R1 |

## License
此代码遵循[OpenSoftware License 1.0](LICENSE)，继承自MIT。

## 联系方式
如果您有任何疑问，请欢迎提issue共同讨论
