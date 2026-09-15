# zero copy
将进程A地址X的物理页映射给进程B地址Y，使得进程A的地址X和进程B的地址Y共享数据，实现数据零拷贝。

## 源码编译
```shell
make clean
make
```
> [支持场景]  
> 支持5.10.0-243.0.0.142.oe2203sp4内核版本编译，对应源码 https://gitee.com/openeuler/kernel/tree/OLK-5.10/ 的提交：3ad50d6b5c4a41571afe95f9da1c1574318c1654。

## 加载驱动
```shell
insmod zcopy.ko
```
