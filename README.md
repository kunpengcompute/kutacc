# Kunpeng Unified Transformer Accelerated Library

## 1.简介
鲲鹏芯片支持向量、矩阵计算，带来算力提升的同时，辅以高速RDMA网络，带来超大带宽、微秒级延迟的极致性能。该芯片强浮点算力和高速带宽天然亲和AI推理计算。基于此，我们提出一种鲲鹏平台上Transformer模型融合算子库（简称“KuTACC”），高效实现Transformer模型推理在鲲鹏处理器的执行。

## 2.本地部署运行

### 2.1 环境依赖搭建

#### 2.1.1 安装 HPCKit 编译套件
该方案是用HPCKit组件中的毕昇编译器进行编译，HPCKit安装流程参考[官方指导文档](https://www.hikunpeng.com/developer/hpc/hpckit-download)。

KuTACC的安装需要使用HPCKit环境中的毕昇编译器、KUPL，配置流程参考[HPCKit介绍](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/devg/KunpengHPCKit_developer_002.html)。

#### 2.1.2 加载毕昇编译器
指定HPCKit安装路径`HPCKIT_PATH`，执行下述命令导入编译器环境变量：
```shell
HPCKIT_PATH=/path-to-HPCKit
source ${HPCKIT_PATH}/latest/compiler/bisheng/env/setvars.sh
export CC=$(which clang)
export CXX=$(which clang++)
export FC=$(which flang)
```

#### 2.1.3 加载KUPL环境变量
指定HPCKit安装路径`HPCKIT_PATH`，执行下述命令导入KUPL环境变量：
```shell
KUPL_PATH=${HPCKIT_PATH}/latest/kupl/bisheng/release
export CPATH=${KUPL_PATH}/include:$CPATH
export LIBRARY_PATH=${KUPL_PATH}/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=${KUPL_PATH}/lib:$LD_LIBRARY_PATH
```

#### 2.1.4 加载ibverbs依赖库变量
通信接口使用了基于verbs接口的RDMA通信功能，执行下述命令配置`libibverbs`库环境变量：
```shell
export LD_LIBRARY_PATH=/usr/lib64/libibverbs:$LD_LIBRARY_PATH
```

#### 2.1.5 加载内核驱动模块文件（可选）
为了进一步释放硬件算力，从底层提升算子推理整体运行效率，可按需加载对应内核模块。

##### 2.1.5.1 zcopy内核驱动模块
zcopy（zero copy）内核驱动依托零拷贝原理，可将不同进程的内存地址相互映射共享，减少传输开销，有效提升数据交互与运算效率。

###### 2.1.5.1.1 内核驱动模块获取
获取zcopy内核驱动模块方式有以下2种：  
（1）openEuler系统获取
若系统内核版本在`5.10.0-294`版本及以上，可直接进入系统默认驱动目录解压使用，下述以`5.10.0-294`版本为例，最终内核驱动为`zcopy.ko`：
```shell
cd /usr/lib/modules/5.10.0-294.0.0.197.oe2203sp4.aarch64/kernel/drivers/misc/zcopy
xz -d zcopy.ko.xz
ll /usr/lib/modules/5.10.0-294.0.0.197.oe2203sp4.aarch64/kernel/drivers/misc/zcopy.ko
```

或者，可在[镜像源下载地址](https://dl-cdn.openeuler.openatom.cn/openEuler-22.03-LTS-SP4/update/aarch64/Packages/)下载，并进行解压提取`zcopy.ko`，下述以`5.10.0-294`版本为例：
```shell
wget https://dl-cdn.openeuler.openatom.cn/openEuler-22.03-LTS-SP4/update/aarch64/Packages/kernel-5.10.0-294.0.0.197.oe2203sp4.aarch64.rpm
rpm2cpio kernel-5.10.0-294.0.0.197.oe2203sp4.aarch64.rpm | cpio -div
cd usr/lib/modules/5.10.0-294.0.0.197.oe2203sp4.aarch64/kernel/drivers/misc/zcopy
xz -d zcopy.ko.xz
ll /usr/lib/modules/5.10.0-294.0.0.197.oe2203sp4.aarch64/kernel/drivers/misc/zcopy.ko
```

（2）其他操作系统  
对于其他操作系统，可以在`/path_to_kutacc/thirdparty/zcopy`下获取zcopy驱动模块源码，进行对应操作系统适配后，按照`README.md`进行编译生成`zcopy.ko`。

###### 2.1.5.1.2 执行命令
获取`zcopy.ko`文件后，执行以下命令完成加载、查看以及卸载:
```shell
insmod zcopy.ko      # 加载内核驱动模块
lsmod | grep zcopy   # 查看内核驱动模块加载状态
rmmod zcopy          # 卸载内核驱动模块
```

##### 2.1.5.2 硬件预取优化内核驱动模块
硬件预取优化驱动模块用于配置硬件预取模式，配合代码中的手动数据预取操作，优化缓存命中率，从而提升计算性能。
若有需要可以联系华为人员。

### 2.2 源码编译与安装
执行下述命令进行源码编译与安装，默认安装路径为`/path_to_kutacc/install`：
```shell
sh build.sh
```

### 2.3 环境变量配置
执行下述命令设置KuTACC的环境变量，而后在目标应用中编译使用即可：
```shell
export KUTACC_PATH=/path_to_kutacc
export KUTACC_LIB=/path_to_kutacc/install/lib
export KUTACC_INCLUDE=/path_to_kutacc/install/include
```

### 2.4 运行测试用例（可选）
测试代码运行依赖于`MVAPICH`，源码获取参考[MVAPICH官网](https://mvapich.cse.ohio-state.edu)，安装流程如下：
```shell
bash script/mvapich_install.sh --prefix=/path/to/install
```

安装完成`MVAPICH`后，执行下述命令加载环境变量：
```shell
export MPI_PATH=/path-to-MVAPICH
export PATH=${MPI_PATH}/bin:$PATH
```

执行下述命令运行测试用例：
```shell
cd test
bash build.sh
bash run_test.sh
```

## 3. 支持的应用

| 支持的应用    |  应用版本 |
|:---------|------:|
| DeepSeek | V3/R1 |
| AlphaFold3 | 鲲鹏社区开源版本 |

## License
此代码遵循[OpenSoftware License 1.0](LICENSE)，继承自MIT。

## 联系方式
如果您有任何疑问，请欢迎提issue共同讨论
