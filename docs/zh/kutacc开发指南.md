# KuTACC开发指南

- **[总述](#总述)**  

- **[KuTACC库函数说明](#kutacc库函数说明)**  

## 总述

KuTACC是基于鲲鹏平台的Transformer模型融合算子库，采用C/C++实现，针对AI算子各层核心算子进行优化，并提供相应接口用于在应用中直接调用，为AI应用提供高效算力支撑。目前KuTACC基于毕昇编译器和libstdc++标准库实现，对gcc编译器暂未支持。

## KuTACC库函数说明

- **[函数说明](#函数说明)**  

- **[函数定义](#函数定义)**  

### 函数说明

本指南以“总述 → KuTACC库函数说明”为文档大纲；函数定义按“接口类型 → 源码模块 → 具体接口”三级组织。接口类型分为通用接口、DeepSeek（DS）接口和 AlphaFold3 接口；模块名称对应 `kunpeng/kutacc/src` 下的目录或源文件，同一父目录下的模块相邻排列。各具体接口依次介绍接口定义、模板参数、参数、返回值和示例。

版本查询函数使用全局 C 链接名称，其余函数及 Tensor 位于 `kutacc` 命名空间，使用 C++ 调用。接口定义保留模板参数和默认参数，省略可见性宏 `kutacc_export`。参数表中的形状、单位及输入/输出方向以头文件和对应实现为依据。`const Tensor&` 限定的是张量描述对象，其底层数据仍可能作为输出写入。

示例中的算子调用片段需放入已准备好数据、缓冲区和通信环境的 C++ 程序；它们不包含完整的模型或分布式初始化流程。使用头文件时可先显式包含其依赖的标准库头文件：

```cpp
#include <array>
#include <optional>
#include <tuple>
#include <utility>
#include "kutacc.h"
```

### 函数定义

- **[通用接口](#通用接口)**

- **[DeepSeek（DS）接口](#deepseekds接口)**

- **[AlphaFold3 接口](#alphafold3-接口)**

#### 通用接口

- **[version（版本信息）](#version版本信息)**

- **[tensor（张量描述）](#tensor张量描述)**

##### version（版本信息）

- **[返回值与版本结构体](#返回值与版本结构体)**

- **[kutacc_get_version](#kutacc_get_version)**

###### 返回值与版本结构体

**返回值定义**

|ReturnCode|类型|值|描述|
|---|---|---|---|
|KUTACC_OK|int|0|执行成功。|
|KUTACC_ERROR|int|-1|执行失败。|

以上状态码用于版本查询等明确约定此返回值的接口；异步线程组创建等接口的返回值以各自说明为准。

**类型定义**

```cpp
typedef struct kutacc_version {
    const char *product_name;
    const char *product_version;
    const char *component_name;
    const char *component_version;
    const char *component_appendinfo;
} kutacc_version_t;
```

|成员|描述|
|---|---|
|product_name|产品名称。|
|product_version|产品版本。|
|component_name|组件名称。|
|component_version|组件版本。|
|component_appendinfo|组件附加信息，例如编译器标识。|

###### kutacc_get_version

获取 KuTACC 产品和组件版本信息。

**接口定义**

```cpp
int kutacc_get_version(kutacc_version_t *version);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|version|`kutacc_version_t *`|产品和组件版本信息结构体指针，不能为空。|输出|

**返回值**

成功返回 `KUTACC_OK`（0）；`version` 为空时返回 `KUTACC_ERROR`（-1）。

**示例**

```cpp
#include <array>
#include <optional>
#include <tuple>
#include <cstdio>
#include "kutacc.h"

int main()
{
    kutacc_version_t version{};
    if (kutacc_get_version(&version) != KUTACC_OK) {
        return 1;
    }
    std::printf("product_name = %s\n", version.product_name);
    std::printf("product_version = %s\n", version.product_version);
    std::printf("component_name = %s\n", version.component_name);
    std::printf("component_version = %s\n", version.component_version);
    std::printf("component_appendinfo = %s\n", version.component_appendinfo);
    return 0;
}
```

**说明：**
>版本信息以实际运行结果为准。

##### tensor（张量描述）

- **[Tensor](#tensor)**

###### Tensor

描述固定维数张量的数据指针、形状和步长。Tensor 不分配或释放底层数据，调用方负责数据内存的生命周期。步长以元素为单位。

**接口定义**

```cpp
template <typename dtype, int64_t dim>
struct Tensor {
    Tensor() = default;
    Tensor(dtype *data_ptr, const int64_t *sizes, const int64_t *strides);
    Tensor(dtype *data_ptr, const std::array<int64_t, dim> &sizes,
        const std::array<int64_t, dim> &strides);
    Tensor(dtype *data_ptr, const int64_t *sizes);
    Tensor(dtype *data_ptr, const std::array<int64_t, dim> &sizes);
    int64_t numel() const;
    const std::array<int64_t, dim> &sizes() const;
    const std::array<int64_t, dim> &strides() const;
    int64_t size(int64_t i) const;
    int64_t stride(int64_t i) const;
    dtype *data_ptr() const;
    void set_data_ptr(dtype *ptr);
    dtype &index(const std::array<int64_t, dim> &indices) const;
    bool is_contiguous() const;
};
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|dtype|模板类型|张量元素类型。|输入|
|dim|`int64_t` 模板参数|张量维数，必须大于 0。|输入|
|data_ptr / ptr|`dtype*`|底层数据指针。|输入|
|sizes|`const int64_t*` 或 `const std::array<int64_t, dim>&`|各维元素数量，包含 dim 项。|输入|
|strides|`const int64_t*` 或 `const std::array<int64_t, dim>&`|各维步长，包含 dim 项；省略时按连续行主序计算。|输入|
|i|`int64_t`|维度索引，合法范围为 `[-dim, dim)`，负索引按 `dim + i` 解析。|输入|
|indices|`const std::array<int64_t, dim>&`|各维元素索引，需在对应维度范围内。|输入|

**返回值**

|成员接口|返回值或行为|
|---|---|
|numel|各维大小的乘积。|
|sizes / strides|形状/步长数组的常量引用。|
|size / stride|指定维度的大小/步长。|
|data_ptr|底层数据指针。|
|set_data_ptr|替换底层指针，不改变形状和步长；无返回值。|
|index|按 `sum(indices[i] * strides[i])` 定位元素并返回可写引用；不检查索引边界。|
|is_contiguous|最后一维步长为 1 且相邻维度满足连续布局关系时返回 true。|

默认构造不会建立有效的数据指针、形状和步长，使用前应赋值为完整构造的 Tensor。

**示例**

```cpp
#include <array>
#include <optional>
#include <tuple>
#include "kutacc.h"

int main()
{
    float data[6] = {0, 1, 2, 3, 4, 5};
    std::array<int64_t, 2> shape = {2, 3};
    kutacc::Tensor<float, 2> tensor(data, shape);
    tensor.index(std::array<int64_t, 2>{1, 2}) = 6.0f;
    return tensor.numel() == 6 && tensor.stride(0) == 3 &&
        tensor.size(-1) == 3 && tensor.is_contiguous() ? 0 : 1;
}
```

#### DeepSeek（DS）接口

- **[attention/flash_mla](#attentionflash_mla)**

- **[attention/flash_mha](#attentionflash_mha)**

- **[core/matmul](#corematmul)**

- **[core/kurmcl](#corekurmcl)**

- **[utils](#utils)**

- **[rmsnorm](#rmsnorm)**

- **[embedding](#embedding)**

- **[moe](#moe)**

- **[pp](#pp)**

- **[comm](#comm)**

- **[mt](#mt)**

- **[async](#async)**

- **[mul_scalar_add](#mul_scalar_add)**

- **[math](#math)**

- **[rope](#rope)**

- **[silu_mul](#silu_mul)**

##### attention/flash_mla

源码模块：`kunpeng/kutacc/src/attention/flash_mla`。

- **[flash_mla_meta_create](#flash_mla_meta_create)**

- **[flash_mla_meta_destory](#flash_mla_meta_destory)**

- **[flash_mla_dense_decode_sched](#flash_mla_dense_decode_sched)**

- **[flash_mla_sparse_decode_sched](#flash_mla_sparse_decode_sched)**

- **[flash_mla_dense_decode](#flash_mla_dense_decode)**

- **[flash_mla_sparse_decode](#flash_mla_sparse_decode)**

相关类型：`typedef struct FlashMLAMeta *FlashMLAMetaHandle;`。使用顺序为 create → dense/sparse sched → 按 `extra_bytes` 分配缓冲区 → 对应 decode → destory。调度与解码的形状、TopK 和打包配置必须一致。

###### flash_mla_meta_create

创建 FlashMLA 元数据句柄，初始化线程调度信息。调用 MLA 调度和解码接口前必须先创建句柄。

**接口定义**

```cpp
void flash_mla_meta_create(FlashMLAMetaHandle &meta);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|meta|`FlashMLAMetaHandle &`|新分配的 FlashMLAMeta 句柄。|输出|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

```cpp
kutacc::FlashMLAMetaHandle meta = nullptr;
kutacc::flash_mla_meta_create(meta);
// 使用 meta 完成调度和解码后释放。
kutacc::flash_mla_meta_destory(meta);
```

###### flash_mla_meta_destory

销毁 FlashMLA 元数据句柄并释放资源。函数名中的 `destory` 与公开声明保持一致。

**接口定义**

```cpp
void flash_mla_meta_destory(FlashMLAMetaHandle meta);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|meta|`FlashMLAMetaHandle`|待销毁的 FlashMLAMeta 句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::flash_mla_meta_destory(meta);
```

###### flash_mla_dense_decode_sched

为稠密 MLA 解码生成分块和线程调度信息，并返回临时缓冲区所需字节数。必须先调度，再分配缓冲区并调用解码接口。

**接口定义**

```cpp
void flash_mla_dense_decode_sched(const Tensor<int, 1> &seqlens_kv, int64_t seqlen_q, int64_t num_heads_q,
    int64_t head_dim, int64_t head_dim_v, int64_t page_block_size, bool is_kv_packed, int64_t &extra_bytes,
    FlashMLAMetaHandle meta);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|seqlens_kv|`const Tensor<int, 1> &`|每批次 KV 序列长度，类型为 `Tensor<int, 1>`。|输入|
|seqlen_q|`int64_t`|Query 序列长度，即查询 token 数。|输入|
|num_heads_q|`int64_t`|Query 注意力头数。|输入|
|head_dim|`int64_t`|Query/Key 头维度，MLA 必须为 576。|输入|
|head_dim_v|`int64_t`|Value 头维度，MLA 必须为 512。|输入|
|page_block_size|`int64_t`|分页 KV 缓存的页块大小，MLA 必须为 64。|输入|
|is_kv_packed|`bool`|为 true 时表示 KV 缓存已打包。|输入|
|extra_bytes|`int64_t &`|额外缓冲区所需字节数，包括 softmax_lse、oaccum 和线程缓冲区。|输出|
|meta|`FlashMLAMetaHandle`|FlashMLA 元数据句柄，调度结果写入该句柄。|输入/输出|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::flash_mla_dense_decode_sched(
    seqlens_kv, seqlen_q, num_heads_q, head_dim, head_dim_v, page_block_size, is_kv_packed, 
    extra_bytes, meta);
```

###### flash_mla_sparse_decode_sched

为稀疏 MLA 解码生成调度信息，计算 TopK 和额外 KV 缓存所需的临时缓冲区字节数。

**接口定义**

```cpp
void flash_mla_sparse_decode_sched(int64_t batch_size, int64_t seqlen_q, int64_t num_heads_q,
    int64_t head_dim, int64_t head_dim_v, int64_t topk, int64_t extra_topk,
    const std::optional<Tensor<int, 1>> &topk_length, const std::optional<Tensor<int, 1>> &extra_topk_length,
    int64_t &extra_bytes, FlashMLAMetaHandle meta);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|batch_size|`int64_t`|批次内序列数。|输入|
|seqlen_q|`int64_t`|Query 序列长度。|输入|
|num_heads_q|`int64_t`|Query 注意力头数。|输入|
|head_dim|`int64_t`|Query/Key 头维度，为 512 或 576。|输入|
|head_dim_v|`int64_t`|Value 头维度，必须为 512。|输入|
|topk|`int64_t`|每个 Query 的稠密部分 TopK 索引数。|输入|
|extra_topk|`int64_t`|额外稀疏部分的 TopK 索引数；没有时为 0。|输入|
|topk_length|`const std::optional<Tensor<int, 1>> &`|可选的逐批次实际稠密 TopK 长度，覆盖 topk。|输入|
|extra_topk_length|`const std::optional<Tensor<int, 1>> &`|可选的逐批次实际稀疏 TopK 长度。|输入|
|extra_bytes|`int64_t &`|额外缓冲区所需字节数。|输出|
|meta|`FlashMLAMetaHandle`|FlashMLA 元数据句柄，调度结果写入该句柄。|输入/输出|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::flash_mla_sparse_decode_sched(
    batch_size, seqlen_q, num_heads_q, head_dim, head_dim_v, topk, extra_topk, topk_length, 
    extra_topk_length, extra_bytes, meta);
```

###### flash_mla_dense_decode

基于分页 KV 缓存执行稠密 MLA 前向解码，使用 split-KV 和在线 Softmax，输出注意力结果及 log-sum-exp。

**接口定义**

```cpp
void flash_mla_dense_decode(const Tensor<bfloat16_t, 4> &q, const Tensor<bfloat16_t, 3> &kcache,
    const std::optional<Tensor<bfloat16_t, 3>> &vcache, const Tensor<int, 2> &block_table,
    const Tensor<int, 1> &seqlens_kv, const Tensor<bfloat16_t, 4> &o, const Tensor<float, 3> &softmax_lse,
    float softmax_scale, bool is_causal, void *extra_buffer, FlashMLAMetaHandle meta);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|q|`const Tensor<bfloat16_t, 4> &`|BF16 Query 张量，形状为 [batch_size, seqlen_q_ori, num_heads_q_ori, head_dim]。|输入|
|kcache|`const Tensor<bfloat16_t, 3> &`|BF16 Key 缓存，形状为 [num_blocks, page_block_size, head_dim]。|输入|
|vcache|`const std::optional<Tensor<bfloat16_t, 3>> &`|可选的 BF16 Value 缓存，形状为 [num_blocks, page_block_size, head_dim_v]；仅 is_kv_packed 为 true 时必需。|输入|
|block_table|`const Tensor<int, 2> &`|int 块表，形状为 [batch_size, max_num_blocks_per_seq]，将序列映射到缓存块。|输入|
|seqlens_kv|`const Tensor<int, 1> &`|每批次实际 KV 序列长度，元素类型为 int。|输入|
|o|`const Tensor<bfloat16_t, 4> &`|BF16 输出张量，形状为 [batch_size, seqlen_q_ori, num_heads_q_ori, head_dim_v]。|输出|
|softmax_lse|`const Tensor<float, 3> &`|在线 Softmax 的 float log-sum-exp 张量，形状为 [batch_size, seqlen_q_ori, num_heads_q_ori]。|输出|
|softmax_scale|`float`|Q·Kᵀ 的缩放因子，通常为 1/sqrt(head_dim)。|输入|
|is_causal|`bool`|为 true 时应用因果掩码。|输入|
|extra_buffer|`void *`|预分配的临时缓冲区，大小由 `flash_mla_dense_decode_sched` 返回。|输入|
|meta|`FlashMLAMetaHandle`|已完成调度的 FlashMLA 元数据句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::flash_mla_dense_decode(
    q, kcache, vcache, block_table, seqlens_kv, o, softmax_lse, softmax_scale, is_causal, 
    extra_buffer, meta);
```

###### flash_mla_sparse_decode

按 TopK 索引选择 KV 缓存执行稀疏 MLA 解码，支持额外 KV 缓存和 attention sink 偏置。

**接口定义**

```cpp
void flash_mla_sparse_decode(const Tensor<bfloat16_t, 4> &q, const Tensor<bfloat16_t, 3> &kvcache,
    const Tensor<int, 3> &indices, const std::optional<Tensor<int, 1>> &topk_length,
    const std::optional<Tensor<bfloat16_t, 3>> &extra_kvcache, const std::optional<Tensor<int, 3>> &extra_indices,
    const std::optional<Tensor<int, 1>> &extra_topk_length, const std::optional<Tensor<float, 1>> &attn_sink,
    const Tensor<bfloat16_t, 4> &o, const Tensor<float, 3> &softmax_lse, float softmax_scale, void *extra_buffer,
    FlashMLAMetaHandle meta);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|q|`const Tensor<bfloat16_t, 4> &`|BF16 Query 张量，形状为 [batch_size, seqlen_q, num_heads_q, head_dim]。|输入|
|kvcache|`const Tensor<bfloat16_t, 3> &`|内存连续的 BF16 KV 缓存，形状为 [num_blocks, page_block_size, head_dim]。|输入|
|indices|`const Tensor<int, 3> &`|int 稠密 TopK 索引，形状为 [batch_size, seqlen_q, topk]，指定参与注意力计算的缓存条目。|输入|
|topk_length|`const std::optional<Tensor<int, 1>> &`|可选的逐批次实际稠密 TopK 条目数，元素类型为 int。|输入|
|extra_kvcache|`const std::optional<Tensor<bfloat16_t, 3>> &`|可选的额外 BF16 KV 缓存，形状为 [extra_num_blocks, extra_page_block_size, head_dim]。|输入|
|extra_indices|`const std::optional<Tensor<int, 3>> &`|可选的额外 int TopK 索引，形状为 [batch_size, seqlen_q, extra_topk]。|输入|
|extra_topk_length|`const std::optional<Tensor<int, 1>> &`|可选的逐批次额外 TopK 实际长度。|输入|
|attn_sink|`const std::optional<Tensor<float, 1>> &`|可选的逐头 float attention sink 偏置，形状为 [num_heads_q]。|输入|
|o|`const Tensor<bfloat16_t, 4> &`|BF16 输出张量，形状为 [batch_size, seqlen_q, num_heads_q, head_dim_v]。|输出|
|softmax_lse|`const Tensor<float, 3> &`|float log-sum-exp 张量，形状为 [batch_size, seqlen_q, num_heads_q]。|输出|
|softmax_scale|`float`|Q·Kᵀ 的缩放因子。|输入|
|extra_buffer|`void *`|预分配的临时缓冲区，大小由 `flash_mla_sparse_decode_sched` 返回。|输入|
|meta|`FlashMLAMetaHandle`|已完成调度的 FlashMLA 元数据句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::flash_mla_sparse_decode(
    q, kvcache, indices, topk_length, extra_kvcache, extra_indices, extra_topk_length, attn_sink, o, 
    softmax_lse, softmax_scale, extra_buffer, meta);
```

##### attention/flash_mha

源码模块：`kunpeng/kutacc/src/attention/flash_mha`。

- **[get_flash_attention_block](#get_flash_attention_block)**

- **[flash_attention_v_block_pack](#flash_attention_v_block_pack)**

- **[flash_attention_k_block_pack](#flash_attention_k_block_pack)**

- **[flash_attention](#flash_attention)**

- **[varlen_attention](#varlen_attention)**

先通过 `get_flash_attention_block` 查询 `br`、`bc`，据此分配线程临时缓冲区及补齐 KV 长度。使用预打包 KV 时，先调用 K/V 打包接口。

###### get_flash_attention_block

查询 Flash Attention 使用的查询块和 KV 块大小。

**接口定义**

```cpp
std::tuple<int64_t, int64_t> get_flash_attention_block();
```

**参数**

无。

**返回值**

`std::tuple<int64_t, int64_t>`，依次为查询块大小 `br` 和 KV 块大小 `bc`。

**示例**

```cpp
auto [br, bc] = kutacc::get_flash_attention_block();
```

###### flash_attention_v_block_pack

将 Value 张量重排为 Flash Attention 使用的分块布局。

**接口定义**

```cpp
void flash_attention_v_block_pack(int64_t kv_len, int64_t num_heads, int64_t vo_head_dim,
    int64_t output_len, int64_t input_stride0, int64_t input_stride1, bfloat16_t *input, bfloat16_t *output);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|kv_len|`int64_t`|Key/Value token 数。|输入|
|num_heads|`int64_t`|注意力头数。|输入|
|vo_head_dim|`int64_t`|Value 头维度，即输出特征大小。|输入|
|output_len|`int64_t`|输出长度，至少为 kv_len 向上补齐到 bc 整数倍后的长度。|输入|
|input_stride0|`int64_t`|输入中相邻 token 的步长，即主维步长。|输入|
|input_stride1|`int64_t`|输入中相邻注意力头的步长。|输入|
|input|`bfloat16_t *`|BF16 Value 输入，形状为 (kv_len, num_heads, vo_head_dim)。|输入|
|output|`bfloat16_t *`|分块顺序的 BF16 Value 打包输出，形状为 (num_heads, padded_len, vo_head_dim)。|输出|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::flash_attention_v_block_pack(
    kv_len, num_heads, vo_head_dim, output_len, input_stride0, input_stride1, input, output);
```

###### flash_attention_k_block_pack

将 Key 张量重排为 Flash Attention 的 Q·Kᵀ 计算所需的分块布局。

**接口定义**

```cpp
void flash_attention_k_block_pack(int64_t kv_len, int64_t num_heads, int64_t qk_head_dim,
    int64_t output_len, int64_t input_stride0, int64_t input_stride1, bfloat16_t *input, bfloat16_t *output);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|kv_len|`int64_t`|Key/Value token 数。|输入|
|num_heads|`int64_t`|注意力头数。|输入|
|qk_head_dim|`int64_t`|Query/Key 头维度，即注意力特征大小。|输入|
|output_len|`int64_t`|输出长度，至少为 kv_len 向上补齐到 bc 整数倍后的长度。|输入|
|input_stride0|`int64_t`|输入中相邻 token 的步长。|输入|
|input_stride1|`int64_t`|输入中相邻注意力头的步长。|输入|
|input|`bfloat16_t *`|BF16 Key 输入，形状为 (kv_len, num_heads, qk_head_dim)。|输入|
|output|`bfloat16_t *`|分块顺序的 BF16 Key 打包输出。|输出|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::flash_attention_k_block_pack(
    kv_len, num_heads, qk_head_dim, output_len, input_stride0, input_stride1, input, output);
```

###### flash_attention

执行基于在线 Softmax 的分块注意力计算，支持因果掩码、变长序列、分块预填充及预打包 KV。

**接口定义**

```cpp
void flash_attention(const Tensor<bfloat16_t, 3> &(q), const Tensor<bfloat16_t, 3> &(k),
    const Tensor<bfloat16_t, 3> &(v), const Tensor<bfloat16_t, 3> &(out), const Tensor<bfloat16_t, 2> &(pack_attn_q),
    const Tensor<bfloat16_t, 3> &(pack_attn_k), const Tensor<bfloat16_t, 3> &(pack_attn_v),
    const Tensor<float, 2> &(attn_s), const Tensor<float, 3> &(attn_out_block_old),
    const Tensor<float, 3> &(attn_out_block_new), const Tensor<float, 2> &(attn_max_block_old),
    const Tensor<float, 2> &(attn_max_block_new), const Tensor<float, 2> &(attn_base_block_old),
    const Tensor<float, 2> &(attn_base_block_new), bool causal, double softmax_scale,
    const Tensor<int, 1> &(query_start_loc), const Tensor<int, 1> &(key_start_loc), int chunked_prefill_size,
    std::vector<int64_t> &seq_lens, std::vector<int64_t> &cur_lens, bool is_kv_packed);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|q|`const Tensor<bfloat16_t, 3> &`|BF16 Query 张量，形状为 (total_q_tokens, num_heads, qk_head_dim)。|输入|
|k|`const Tensor<bfloat16_t, 3> &`|BF16 Key 张量，形状为 (total_kv_tokens, num_heads, qk_head_dim)。|输入|
|v|`const Tensor<bfloat16_t, 3> &`|BF16 Value 张量，形状为 (total_kv_tokens, num_heads, vo_head_dim)。|输入|
|out|`const Tensor<bfloat16_t, 3> &`|BF16 输出张量，形状为 (total_q_tokens, num_heads, vo_head_dim)。|输出|
|pack_attn_q|`const Tensor<bfloat16_t, 2> &`|Query 打包缓冲区，形状为 (num_threads, br * qk_head_dim)。|输入|
|pack_attn_k|`const Tensor<bfloat16_t, 3> &`|预打包 Key；单序列且 is_kv_packed 为 true 时，形状为 (num_heads, padded_k_len, qk_head_dim)，否则不使用。|输入|
|pack_attn_v|`const Tensor<bfloat16_t, 3> &`|预打包 Value，布局与 pack_attn_k 类似。|输入|
|attn_s|`const Tensor<float, 2> &`|各线程的 float 分数临时缓冲区，形状为 (num_threads, br * bc)，br、bc 为块大小。|输入|
|attn_out_block_old|`const Tensor<float, 3> &`|各线程的 float 输出临时缓冲区，形状为 (num_threads, br, vo_head_dim)。|输入|
|attn_out_block_new|`const Tensor<float, 3> &`|用于交替计算的第二份 float 输出临时缓冲区。|输入|
|attn_max_block_old|`const Tensor<float, 2> &`|各线程的最大值临时缓冲区，形状为 (num_threads, br)。|输入|
|attn_max_block_new|`const Tensor<float, 2> &`|第二份最大值临时缓冲区。|输入|
|attn_base_block_old|`const Tensor<float, 2> &`|各线程的指数和临时缓冲区，形状为 (num_threads, br)。|输入|
|attn_base_block_new|`const Tensor<float, 2> &`|第二份指数和临时缓冲区。|输入|
|causal|`bool`|为 true 时应用因果掩码，禁止关注后续 token。|输入|
|softmax_scale|`double`|Softmax 前应用于 Q·Kᵀ 的缩放因子，通常为 1/sqrt(head_dim)。|输入|
|query_start_loc|`const Tensor<int, 1> &`|Query 序列累计长度，包含 num_seqs + 1 项。|输入|
|key_start_loc|`const Tensor<int, 1> &`|Key 序列累计长度，包含 num_seqs + 1 项。|输入|
|chunked_prefill_size|`int`|分块预填充的块大小，用于增量解码。|输入|
|seq_lens|`std::vector<int64_t> &`|各序列的当前总长度，在预填充过程中更新。|输入/输出|
|cur_lens|`std::vector<int64_t> &`|各序列已处理的长度，在预填充过程中更新。|输入/输出|
|is_kv_packed|`bool`|仅单序列且 KV 已预打包时设为 true。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::flash_attention(
    q, k, v, out, pack_attn_q, pack_attn_k, pack_attn_v, attn_s, attn_out_block_old, 
    attn_out_block_new, attn_max_block_old, attn_max_block_new, attn_base_block_old, 
    attn_base_block_new, causal, softmax_scale, query_start_loc, key_start_loc, chunked_prefill_size, 
    seq_lens, cur_lens, is_kv_packed);
```

###### varlen_attention

根据序列累计起始位置处理拼接的变长 Q、K、V，执行注意力计算。

**接口定义**

```cpp
void varlen_attention(const Tensor<bfloat16_t, 3> &(q), const Tensor<bfloat16_t, 3> &(k),
    const Tensor<bfloat16_t, 3> &(v), const Tensor<bfloat16_t, 3> &(out), bool causal, double softmax_scale,
    const Tensor<int, 1> &(query_start_loc), const Tensor<int, 1> &(key_start_loc));
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|q|`const Tensor<bfloat16_t, 3> &`|BF16 Query 张量，形状为 (total_q_tokens, num_heads, qk_head_dim)。|输入|
|k|`const Tensor<bfloat16_t, 3> &`|BF16 Key 张量，形状为 (total_kv_tokens, num_heads, qk_head_dim)。|输入|
|v|`const Tensor<bfloat16_t, 3> &`|BF16 Value 张量，形状为 (total_kv_tokens, num_heads, vo_head_dim)。|输入|
|out|`const Tensor<bfloat16_t, 3> &`|BF16 输出张量，形状为 (total_q_tokens, num_heads, vo_head_dim)。|输出|
|causal|`bool`|为 true 时应用因果掩码。|输入|
|softmax_scale|`double`|Q·Kᵀ 的缩放因子。|输入|
|query_start_loc|`const Tensor<int, 1> &`|Query 序列累计长度，包含 num_seqs + 1 项。|输入|
|key_start_loc|`const Tensor<int, 1> &`|Key 序列累计长度，包含 num_seqs + 1 项。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::varlen_attention(q, k, v, out, causal, softmax_scale, query_start_loc, key_start_loc);
```

##### core/matmul

源码模块：`kunpeng/kutacc/src/core/matmul`。

- **[get_s8_s8_gemm_bf16_dq_tmpc_size](#get_s8_s8_gemm_bf16_dq_tmpc_size)**

- **[s8_s8_gemm_bf16_dq](#s8_s8_gemm_bf16_dq)**

- **[s8_s8_packed_gemm_bf16_dq](#s8_s8_packed_gemm_bf16_dq)**

- **[s8_gemm_pack](#s8_gemm_pack)**

- **[get_fusedmoe_s8gemm_workspace_size](#get_fusedmoe_s8gemm_workspace_size)**

- **[s8_gemm_pack_fusedmoe](#s8_gemm_pack_fusedmoe)**

- **[fusedmoe_gateup](#fusedmoe_gateup)**

- **[fusedmoe_down](#fusedmoe_down)**

- **[bf16_packed_gemm](#bf16_packed_gemm)**

- **[bf16_gemm_pack](#bf16_gemm_pack)**

- **[batch_bf16_gemm_pack](#batch_bf16_gemm_pack)**

- **[batch_bf16_s8_packed_gemm_bf16](#batch_bf16_s8_packed_gemm_bf16)**

相关类型：`using MatrixTilingBlock = std::tuple<int64_t, int64_t, int64_t>;`，依次表示 `(tile_m, tile_n, tile_k)`。INT8 GEMM 要求 M/N/K 分别为对应块大小的整数倍，且 `K % 4 == 0`、`tile_k % 4 == 0`；充分利用线程时应满足 `(M / tile_m) * (N / tile_n) * (K / tile_k) == thread_num`。打包和计算使用兼容的分块参数，临时空间按对应查询接口的单位分配。

###### get_s8_s8_gemm_bf16_dq_tmpc_size

查询 INT8 GEMM 在 K 维分块时所需的中间累加缓冲区字节数。

**接口定义**

```cpp
int64_t get_s8_s8_gemm_bf16_dq_tmpc_size(int64_t m, int64_t n, int64_t k, MatrixTilingBlock t);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|m|`int64_t`|左矩阵行数 M。|输入|
|n|`int64_t`|右矩阵对应的输出列数 N。|输入|
|k|`int64_t`|收缩维度 K。|输入|
|t|`MatrixTilingBlock`|分块大小 (tile_m, tile_n, tile_k)，本接口仅使用 tile_k。|输入|

**返回值**

`int64_t`，所需临时缓冲区大小，单位为字节。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
auto result = kutacc::get_s8_s8_gemm_bf16_dq_tmpc_size(m, n, k, t);
```

###### s8_s8_gemm_bf16_dq

打包左侧 INT8 激活矩阵，执行 INT8 矩阵乘法并反量化为 BF16 输出。

**接口定义**

```cpp
void s8_s8_gemm_bf16_dq(int64_t m, int64_t n, int64_t k, MatrixTilingBlock t, int8_t *act_ptr,
    int8_t *input_ptr, int8_t *weight_ptr, float *act_scale_ptr, float *weight_scale_ptr, bfloat16_t *output_ptr,
    bfloat16_t *workspace);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|m|`int64_t`|左矩阵行数 M。|输入|
|n|`int64_t`|右矩阵对应的输出列数 N。|输入|
|k|`int64_t`|收缩维度 K。|输入|
|t|`MatrixTilingBlock`|分块大小 (tile_m, tile_n, tile_k)，必须满足 tile_m == m。|输入|
|act_ptr|`int8_t *`|行主序 INT8 激活矩阵，形状为 (m, k)。|输入|
|input_ptr|`int8_t *`|保存打包左矩阵的临时缓冲区，大小满足打包布局要求。|输出|
|weight_ptr|`int8_t *`|行主序 INT8 权重矩阵，形状为 (n, k)。|输入|
|act_scale_ptr|`float *`|逐行 float 激活缩放因子，长度为 m。|输入|
|weight_scale_ptr|`float *`|逐行 float 权重缩放因子，长度为 n。|输入|
|output_ptr|`bfloat16_t *`|行主序 BF16 输出矩阵，形状为 (m, n)。|输出|
|workspace|`bfloat16_t *`|临时工作缓冲区，大小由 `get_s8_s8_gemm_bf16_dq_tmpc_size` 查询。|输出|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::s8_s8_gemm_bf16_dq(
    m, n, k, t, act_ptr, input_ptr, weight_ptr, act_scale_ptr, weight_scale_ptr, output_ptr, 
    workspace);
```

###### s8_s8_packed_gemm_bf16_dq

使用已打包的 INT8 左右矩阵执行矩阵乘法，并根据缩放因子反量化为 BF16。

**接口定义**

```cpp
void s8_s8_packed_gemm_bf16_dq(int64_t m, int64_t n, int64_t k, MatrixTilingBlock t, int8_t *act_ptr,
    int8_t *weight_ptr, float *act_scale_ptr, float *weight_scale_ptr, bfloat16_t *output_ptr, bfloat16_t *workspace);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|m|`int64_t`|行数 M。|输入|
|n|`int64_t`|列数 N。|输入|
|k|`int64_t`|收缩维度 K。|输入|
|t|`MatrixTilingBlock`|分块大小 (tile_m, tile_n, tile_k)。|输入|
|act_ptr|`int8_t *`|已打包的 INT8 左矩阵，逻辑形状为 (m, k)。|输入|
|weight_ptr|`int8_t *`|已打包的 INT8 右矩阵，逻辑形状为 (n, k)。|输入|
|act_scale_ptr|`float *`|逐行 float 激活缩放因子，长度为 m。|输入|
|weight_scale_ptr|`float *`|逐行 float 权重缩放因子，长度为 n。|输入|
|output_ptr|`bfloat16_t *`|行主序 BF16 输出矩阵，形状为 (m, n)。|输出|
|workspace|`bfloat16_t *`|临时工作缓冲区。|输出|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::s8_s8_packed_gemm_bf16_dq(
    m, n, k, t, act_ptr, weight_ptr, act_scale_ptr, weight_scale_ptr, output_ptr, workspace);
```

###### s8_gemm_pack

按指定行列块大小，将 INT8 矩阵打包为适用于 SME 的布局；支持通过索引数组选取行。

**接口定义**

```cpp
void s8_gemm_pack(int64_t r, int64_t c, int64_t split_r, int64_t split_c, int8_t *input_ptr,
    int8_t *output_ptr, int64_t ldc = 0, bool with_idx = 0, int *idx = nullptr);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|r|`int64_t`|输入矩阵行数。|输入|
|c|`int64_t`|输入矩阵列数。|输入|
|split_r|`int64_t`|打包行块大小，建议为 16 的整数倍以获得最佳性能。|输入|
|split_c|`int64_t`|打包列块大小，建议为 64 的整数倍以获得最佳性能。|输入|
|input_ptr|`int8_t *`|行主序 INT8 输入矩阵。|输入|
|output_ptr|`int8_t *`|按指定布局打包的输出矩阵。|输出|
|ldc|`int64_t`|input_ptr 的主维步长；为 0 时按 c 处理。 默认值：`0`。|输入|
|with_idx|`bool`|为 true 时通过 idx 数组进行索引打包，用于融合 MoE。 默认值：`0`。|输入|
|idx|`int *`|可选的 int 索引数组，长度为 r，将每行映射到 token ID。 默认值：`nullptr`。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::s8_gemm_pack(r, c, split_r, split_c, input_ptr, output_ptr, ldc, with_idx, idx);
```

###### get_fusedmoe_s8gemm_workspace_size

查询融合 MoE INT8 GEMM 所需的激活打包、累加和缩放临时缓冲区大小。

**接口定义**

```cpp
std::tuple<int64_t, int64_t, int64_t> get_fusedmoe_s8gemm_workspace_size(
    int64_t N, int64_t K, int64_t tilebuf_size, MatrixTilingBlock t);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|N|`int64_t`|输出特征数，即权重矩阵对应的输出列数。|输入|
|K|`int64_t`|收缩维度，即公共特征大小。|输入|
|tilebuf_size|`int64_t`|单批次最多处理的 token 数，即 M 维块大小。|输入|
|t|`MatrixTilingBlock`|分块大小 (tile_m, tile_n, tile_k)，本接口仅使用 tile_k。|输入|

**返回值**

`std::tuple<int64_t, int64_t, int64_t>`，依次为 `(tmpx_size, tmpy_size, tmpscale_size)`，单位为各缓冲区的元素个数，不是字节数。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
auto [tmpx_size, tmpy_size, tmpscale_size] = kutacc::get_fusedmoe_s8gemm_workspace_size(
    N, K, tilebuf_size, t);
```

###### s8_gemm_pack_fusedmoe

为融合 MoE 打包激活矩阵切片，支持按 token 索引取数。

**接口定义**

```cpp
void s8_gemm_pack_fusedmoe(int64_t r, int64_t c, int64_t split_c, int8_t *input_ptr,
    int8_t *output_ptr, int64_t ldc = 0, bool with_idx = 0, int *idx = nullptr);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|r|`int64_t`|待打包的行数，即 token 数。|输入|
|c|`int64_t`|列数 K。|输入|
|split_c|`int64_t`|打包列块大小 tile_k。|输入|
|input_ptr|`int8_t *`|行主序 INT8 激活矩阵。|输入|
|output_ptr|`int8_t *`|打包输出缓冲区。|输出|
|ldc|`int64_t`|input_ptr 的主维步长；为 0 时按 c 处理。 默认值：`0`。|输入|
|with_idx|`bool`|为 true 时使用 idx 进行索引打包。 默认值：`0`。|输入|
|idx|`int *`|token ID 索引数组，长度为 r。 默认值：`nullptr`。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::s8_gemm_pack_fusedmoe(r, c, split_c, input_ptr, output_ptr, ldc, with_idx, idx);
```

###### fusedmoe_gateup

执行融合 MoE 的 gate/up 投影，结合专家 token 索引、缩放、INT8 GEMM 和反量化处理。

**接口定义**

```cpp
void fusedmoe_gateup(int64_t total_bs, int64_t K, int64_t N, int64_t num_experts, int64_t lda,
    int64_t ldas, int8_t *acts, int8_t *weights, float *acts_scale, float *weights_scale, int *token_ids,
    int *experts_offset, bfloat16_t *output, int8_t *tmpx, float *tmpy, float *tmp_scales, MatrixTilingBlock t,
    int64_t tilebuf_size, std::optional<int64_t> n_slice);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|total_bs|`int64_t`|所有专家的 token 总数。|输入|
|K|`int64_t`|收缩维度，即特征大小。|输入|
|N|`int64_t`|输出特征大小，即 gate/up 的中间维度。|输入|
|num_experts|`int64_t`|专家数。|输入|
|lda|`int64_t`|输入激活矩阵的主维步长 K。|输入|
|ldas|`int64_t`|激活缩放数组的主维步长，即相邻 token 缩放数据的间距。|输入|
|acts|`int8_t *`|INT8 激活矩阵，每个 token 的特征维度为 K，按 token_ids 索引。|输入|
|weights|`int8_t *`|INT8 权重矩阵，形状为 (num_experts * N, K)。|输入|
|acts_scale|`float *`|逐 token 的 float 激活缩放因子，长度为 total_bs。|输入|
|weights_scale|`float *`|逐专家的 float 权重缩放因子，形状为 (num_experts, N)。|输入|
|token_ids|`int *`|int token ID 映射，长度为 total_bs，用于获取对应的激活缩放因子。|输入|
|experts_offset|`int *`|各专家在 token_ids 中的偏移，长度为 num_experts + 1。|输入|
|output|`bfloat16_t *`|BF16 输出矩阵，形状为 (total_bs, N)。|输出|
|tmpx|`int8_t *`|激活打包临时缓冲区，大小由 `get_fusedmoe_s8gemm_workspace_size` 查询。|输出|
|tmpy|`float *`|float 累加临时缓冲区。|输出|
|tmp_scales|`float *`|float 缩放临时缓冲区。|输出|
|t|`MatrixTilingBlock`|分块大小 (tile_m, tile_n, tile_k)。|输入|
|tilebuf_size|`int64_t`|单块最多处理的 token 数，用于限制缓冲区需求。|输入|
|n_slice|`std::optional<int64_t>`|可选的 N 维切片参数，用于双专家并行优化。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::fusedmoe_gateup(
    total_bs, K, N, num_experts, lda, ldas, acts, weights, acts_scale, weights_scale, token_ids, 
    experts_offset, output, tmpx, tmpy, tmp_scales, t, tilebuf_size, n_slice);
```

###### fusedmoe_down

执行融合 MoE 的 down 投影，使用 INT8 GEMM 将中间特征投影回输出特征维度。

**接口定义**

```cpp
void fusedmoe_down(int64_t total_bs, int64_t K, int64_t N, int64_t num_experts, int8_t *acts,
    int8_t *weights, float *acts_scale, float *weights_scale, int *experts_offset, bfloat16_t *output, int8_t *tmpx,
    float *tmpy, MatrixTilingBlock t, int64_t tilebuf_size, std::optional<int64_t> n_slice);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|total_bs|`int64_t`|所有专家的 token 总数。|输入|
|K|`int64_t`|收缩维度，即 gate/up 输出的特征大小。|输入|
|N|`int64_t`|输出特征大小，即最终隐藏维度。|输入|
|num_experts|`int64_t`|专家数。|输入|
|acts|`int8_t *`|来自 gate/up 且已处理的 INT8 激活矩阵。|输入|
|weights|`int8_t *`|INT8 权重矩阵，形状为 (num_experts * N, K)。|输入|
|acts_scale|`float *`|逐 token 的 float 激活缩放因子，长度为 total_bs。|输入|
|weights_scale|`float *`|逐专家的 float 权重缩放因子，形状为 (num_experts, N)。|输入|
|experts_offset|`int *`|各专家的 token 索引偏移。|输入|
|output|`bfloat16_t *`|BF16 输出矩阵，形状为 (total_bs, N)。|输出|
|tmpx|`int8_t *`|激活打包临时缓冲区。|输出|
|tmpy|`float *`|float 累加临时缓冲区。|输出|
|t|`MatrixTilingBlock`|矩阵分块大小。|输入|
|tilebuf_size|`int64_t`|每块最多处理的 token 数。|输入|
|n_slice|`std::optional<int64_t>`|可选的 N 维切片参数，用于双专家并行优化。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::fusedmoe_down(
    total_bs, K, N, num_experts, acts, weights, acts_scale, weights_scale, experts_offset, output, 
    tmpx, tmpy, t, tilebuf_size, n_slice);
```

###### bf16_packed_gemm

使用已打包的 BF16 左右矩阵执行矩阵乘法，支持可选的逐行或逐列偏置。

**接口定义**

```cpp
void bf16_packed_gemm(int64_t m, int64_t n, int64_t k, MatrixTilingBlock t, bfloat16_t *act_ptr,
    bfloat16_t *weight_ptr, bfloat16_t *output_ptr, bfloat16_t *tmpc, float *bias = nullptr, bool row_bias = false);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|m|`int64_t`|左矩阵行数 M。|输入|
|n|`int64_t`|右矩阵对应的输出列数 N。|输入|
|k|`int64_t`|收缩维度 K。|输入|
|t|`MatrixTilingBlock`|分块大小 (tile_m, tile_n, tile_k)。|输入|
|act_ptr|`bfloat16_t *`|已打包的 BF16 左矩阵，逻辑形状为 (m, k)。|输入|
|weight_ptr|`bfloat16_t *`|已打包的 BF16 右矩阵，逻辑形状为 (n, k)。|输入|
|output_ptr|`bfloat16_t *`|行主序 BF16 输出矩阵，形状为 (m, n)。|输出|
|tmpc|`bfloat16_t *`|K 维分块时使用的临时工作缓冲区。|输出|
|bias|`float *`|可选的偏置缓冲区。 默认值：`nullptr`。|输入|
|row_bias|`bool`|偏置类型：true 表示逐行偏置，false 表示逐列偏置。 默认值：`false`。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::bf16_packed_gemm(m, n, k, t, act_ptr, weight_ptr, output_ptr, tmpc, bias, row_bias);
```

###### bf16_gemm_pack

按指定行列块大小，将 BF16 矩阵打包为适用于 SME 的布局。

**接口定义**

```cpp
void bf16_gemm_pack(int64_t r, int64_t c, int64_t split_r, int64_t split_c, bfloat16_t *input_ptr,
    bfloat16_t *output_ptr);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|r|`int64_t`|行数。|输入|
|c|`int64_t`|列数。|输入|
|split_r|`int64_t`|行块大小，应为 16 的整数倍。|输入|
|split_c|`int64_t`|列块大小，应为 64 的整数倍。|输入|
|input_ptr|`bfloat16_t *`|行主序 BF16 输入矩阵。|输入|
|output_ptr|`bfloat16_t *`|打包输出矩阵。|输出|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::bf16_gemm_pack(r, c, split_r, split_c, input_ptr, output_ptr);
```

###### batch_bf16_gemm_pack

批量打包矩阵。`dtype` 选择单字节或双字节元素对应的打包路径，具体含义见参数表。

**接口定义**

```cpp
void batch_bf16_gemm_pack(int64_t bs, int64_t m, int64_t n, int64_t stride_bs, int64_t stride_m,
    void *src, void *dst, int64_t dtype);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|bs|`int64_t`|批次大小。|输入|
|m|`int64_t`|每个矩阵的行数。|输入|
|n|`int64_t`|每个矩阵的列数。|输入|
|stride_bs|`int64_t`|源缓冲区中相邻矩阵的步长，单位为元素。|输入|
|stride_m|`int64_t`|源缓冲区中矩阵相邻行的步长。|输入|
|src|`void *`|源缓冲区，实际元素类型由 dtype 决定。|输入|
|dst|`void *`|目标打包缓冲区。|输出|
|dtype|`int64_t`|元素字节数及打包路径选择：1 对应 INT8，使用 A 矩阵转置打包；2 对应 BF16，使用 B 矩阵非转置打包。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::batch_bf16_gemm_pack(bs, m, n, stride_bs, stride_m, src, dst, dtype);
```

###### batch_bf16_s8_packed_gemm_bf16

批量执行 BF16 激活与 INT8 权重的混合精度矩阵乘法，输出 BF16，并支持缩放因子。

**接口定义**

```cpp
void batch_bf16_s8_packed_gemm_bf16(int64_t bs, int64_t m, int64_t n, int64_t k, int64_t stride_bs,
    int64_t stride_m, bfloat16_t *act, int8_t *weight, bfloat16_t *out, float *rscale, float *cscale);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|bs|`int64_t`|批次大小。|输入|
|m|`int64_t`|A 和 C 的行数，即每批次 M。|输入|
|n|`int64_t`|B 和 C 对应的输出列数，即每批次 N。|输入|
|k|`int64_t`|收缩维度 K。|输入|
|stride_bs|`int64_t`|A 和 C 的批次步长，单位为元素。|输入|
|stride_m|`int64_t`|批次内相邻行的步长，单位为元素。|输入|
|act|`bfloat16_t *`|BF16 激活张量，形状为 (bs, m, k)。|输入|
|weight|`int8_t *`|INT8 权重张量，形状为 (bs, n, k)。|输入|
|out|`bfloat16_t *`|BF16 输出张量，形状为 (bs, m, n)。|输出|
|rscale|`float *`|可选的逐批次 float 缩放因子，形状为 (bs, n)。|输入|
|cscale|`float *`|可选的逐批次 float 缩放因子，形状为 (bs, k)。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::batch_bf16_s8_packed_gemm_bf16(bs, m, n, k, stride_bs, stride_m, act, weight, out, rscale, cscale);
```

##### core/kurmcl

源码模块：`kunpeng/kutacc/src/core/kurmcl`。

- **[kurmcl_comm_create](#kurmcl_comm_create)**

- **[kurmcl_barrier](#kurmcl_barrier)**

**相关类型**

```cpp
typedef enum kurmcl_datatype {
    KURMCL_DATATYPE_CHAR = 0,
    KURMCL_DATATYPE_INT,
    KURMCL_DATATYPE_LONG,
    KURMCL_DATATYPE_FLOAT,
    KURMCL_DATATYPE_DOUBLE
} kurmcl_datatype_t;

typedef int (*kurmcl_oob_allgather_cb_t)(
    const void *sendbuf, void *recvbuf, int count, void *group, kurmcl_datatype_t datatype);

typedef int (*kurmcl_oob_barrier_cb_t)(void *group);

typedef int (*kurmcl_oob_alltoall_cb_t)(const void *sendbuf, int sendcount, kurmcl_datatype_t sendtype, void *recvbuf,
    int recvcount, kurmcl_datatype_t recvtype, void *group);

typedef struct kurmcl_oob_cb {
    kurmcl_oob_allgather_cb_t oob_allgather;
    kurmcl_oob_barrier_cb_t oob_barrier;
    kurmcl_oob_alltoall_cb_t oob_alltoall;
} kurmcl_oob_cb_t, *kurmcl_oob_cb_h;

typedef struct kurmcl_conn_info *kurmcl_conn_info_h;
```

带外回调成功返回 0，失败返回非 0；`group` 为调用方传入的通信组上下文，`recvbuf` 为接收输出缓冲区。

###### kurmcl_comm_create

使用带外通信回调创建 KuRMCL 通信器。

**接口定义**

```cpp
int kurmcl_comm_create(int size, int rank, kurmcl_oob_cb_h oob_cbs, void *group,
    kurmcl_conn_info_h *conn_info);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|size|`int`|通信器的进程数。|输入|
|rank|`int`|当前进程在通信器中的编号。|输入|
|oob_cbs|`kurmcl_oob_cb_h`|带外通信回调集合。|输入|
|group|`void *`|MPI 通信器上下文。|输入|
|conn_info|`kurmcl_conn_info_h *`|KuRMCL 通信器句柄。|输出|

**返回值**

成功返回 0，失败返回 -1。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
auto result = kutacc::kurmcl_comm_create(size, rank, oob_cbs, group, conn_info);
```

###### kurmcl_barrier

在指定 KuRMCL 通信器上执行同步屏障。

**接口定义**

```cpp
void kurmcl_barrier(kurmcl_conn_info_h conn_info);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|conn_info|`kurmcl_conn_info_h`|KuRMCL 通信器句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::kurmcl_barrier(conn_info);
```

##### utils

源码模块：`kunpeng/kutacc/src/utils`。

- **[quant](#quant)**

- **[quant_pack](#quant_pack)**

- **[quant_save_pack](#quant_save_pack)**

###### quant

按行计算绝对最大值及缩放因子，将 BF16 矩阵量化为 INT8，输出每行缩放因子。

**接口定义**

```cpp
void quant(int64_t height, int64_t width, const bfloat16_t *input, int64_t input_stride, int8_t *out,
    int64_t out_stride, float *scale);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|height|`int64_t`|行数。|输入|
|width|`int64_t`|列数。|输入|
|input|`const bfloat16_t *`|BF16 输入矩阵。|输入|
|input_stride|`int64_t`|input 的行步长。|输入|
|out|`int8_t *`|INT8 输出矩阵。|输出|
|out_stride|`int64_t`|out 的行步长。|输入|
|scale|`float *`|逐行 float 缩放因子，数组长度为 height。|输出|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::quant(height, width, input, input_stride, out, out_stride, scale);
```

###### quant_pack

将 BF16 矩阵按行量化为 INT8，并输出打包布局和缩放因子。

**接口定义**

```cpp
void quant_pack(int64_t height, int64_t width, bfloat16_t *input_data, int8_t *output_data,
    float *scale_data);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|height|`int64_t`|行数。|输入|
|width|`int64_t`|列数。|输入|
|input_data|`bfloat16_t *`|BF16 输入矩阵。|输入|
|output_data|`int8_t *`|按指定布局打包的 INT8 输出数据。|输出|
|scale_data|`float *`|逐行 float 缩放因子，数组长度为 height。|输出|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::quant_pack(height, width, input_data, output_data, scale_data);
```

###### quant_save_pack

量化 BF16 矩阵，同时输出普通行主序和打包布局的 INT8 数据及对应缩放因子。

**接口定义**

```cpp
void quant_save_pack(int64_t height, int64_t width, bfloat16_t *input_data, int64_t ldo,
    int8_t *output_data, int64_t lds, float *scale_data, int8_t *packed_output_data, float *packed_scale_data);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|height|`int64_t`|行数。|输入|
|width|`int64_t`|列数。|输入|
|input_data|`bfloat16_t *`|BF16 输入矩阵。|输入|
|ldo|`int64_t`|output_data 的主维步长，即行步长。|输入|
|output_data|`int8_t *`|行主序 INT8 输出矩阵。|输出|
|lds|`int64_t`|scale_data 的主维步长，即行步长。|输入|
|scale_data|`float *`|逐行 float 缩放因子，按 lds 步长存储。|输出|
|packed_output_data|`int8_t *`|打包布局的 INT8 输出数据。|输出|
|packed_scale_data|`float *`|与打包布局对应的逐行缩放因子。|输出|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::quant_save_pack(
    height, width, input_data, ldo, output_data, lds, scale_data, packed_output_data, 
    packed_scale_data);
```

##### rmsnorm

源码模块：`kunpeng/kutacc/src/rmsnorm`。

- **[rmsnorm](#rmsnorm-1)**

- **[rmsnorm_quant](#rmsnorm_quant)**

###### rmsnorm

执行 RMS 归一化和权重缩放；模板参数控制是否在归一化前融合残差。

**接口定义**

```cpp
template <bool has_residual>
void rmsnorm(int64_t height, int64_t width, bfloat16_t *acts, int64_t acts_stride,
    const bfloat16_t *weights, float eps, bfloat16_t *residual, bfloat16_t *outs);
```

**模板参数**

|参数名|描述|
|---|---|
|has_residual|为 true 时在归一化前累加残差，为 false 时不使用残差。|

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|height|`int64_t`|行数。|输入|
|width|`int64_t`|每行的列数。|输入|
|acts|`bfloat16_t *`|BF16 激活矩阵，行步长为 acts_stride。|输入|
|acts_stride|`int64_t`|acts 的行步长。|输入|
|weights|`const bfloat16_t *`|RMS 权重向量，长度为 width。|输入|
|eps|`float`|防止除零的数值稳定项 epsilon。|输入|
|residual|`bfloat16_t *`|BF16 残差矩阵，与激活形状相同；has_residual 为 true 时写回累加后的残差。|输入/输出|
|outs|`bfloat16_t *`|行主序 BF16 归一化输出矩阵。|输出|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::rmsnorm<false>(height, width, acts, acts_stride, weights, eps, residual, outs);
```

###### rmsnorm_quant

融合 RMS 归一化与逐行 INT8 量化，输出量化结果和动态缩放因子。

**接口定义**

```cpp
template <bool has_residual>
void rmsnorm_quant(int64_t height, int64_t width, bfloat16_t *acts, int64_t acts_stride,
    const bfloat16_t *weights, float eps, bfloat16_t *residual, int64_t res_stride, int8_t *outs, int64_t outs_stride,
    float *scales, int64_t scales_stride);
```

**模板参数**

|参数名|描述|
|---|---|
|has_residual|为 true 时在归一化前累加残差，为 false 时不使用残差。|

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|height|`int64_t`|行数。|输入|
|width|`int64_t`|每行的列数。|输入|
|acts|`bfloat16_t *`|BF16 激活矩阵，行步长为 acts_stride。|输入|
|acts_stride|`int64_t`|acts 的步长。|输入|
|weights|`const bfloat16_t *`|RMS 权重向量。|输入|
|eps|`float`|RMS 计算的数值稳定项 epsilon。|输入|
|residual|`bfloat16_t *`|BF16 残差矩阵，步长为 res_stride；has_residual 为 true 时写回累加后的残差。|输入/输出|
|res_stride|`int64_t`|residual 的步长。|输入|
|outs|`int8_t *`|INT8 输出矩阵，步长为 outs_stride。|输出|
|outs_stride|`int64_t`|outs 的步长。|输入|
|scales|`float *`|逐行缩放因子，步长为 scales_stride。|输出|
|scales_stride|`int64_t`|scales 的步长。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::rmsnorm_quant<false>(
    height, width, acts, acts_stride, weights, eps, residual, res_stride, outs, outs_stride, scales, 
    scales_stride);
```

##### embedding

源码模块：`kunpeng/kutacc/src/embedding`。

- **[embedding](#embedding-1)**

###### embedding

根据 token 索引查询嵌入表，支持通过词表起止索引限定有效词表范围。

**接口定义**

```cpp
void embedding(const int64_t *input, const void *weight, void *out, int64_t element_size,
    int64_t n_tokens, int64_t hidden, int64_t vocab_start, int64_t vocab_end);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|input|`const int64_t *`|int64_t token 索引数组，长度为 n_tokens。|输入|
|weight|`const void *`|嵌入表，每行占 element_size * hidden 字节。|输入|
|out|`void *`|输出缓冲区，形状为 (n_tokens, hidden)。|输出|
|element_size|`int64_t`|每个嵌入元素的字节数，例如 BF16 为 2。|输入|
|n_tokens|`int64_t`|待查询的 token 数。|输入|
|hidden|`int64_t`|隐藏维度大小。|输入|
|vocab_start|`int64_t`|有效词表范围的起始索引，包含该索引。|输入|
|vocab_end|`int64_t`|有效词表范围的结束索引，不包含该索引。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::embedding(input, weight, out, element_size, n_tokens, hidden, vocab_start, vocab_end);
```

##### moe

源码模块：`kunpeng/kutacc/src/moe`。

- **[moe_dispatch_init](#moe_dispatch_init)**

- **[moe_dispatch_send](#moe_dispatch_send)**

- **[moe_dispatch_recv](#moe_dispatch_recv)**

- **[moe_dispatch_finalize](#moe_dispatch_finalize)**

- **[moe_combine_init](#moe_combine_init)**

- **[moe_combine_send](#moe_combine_send)**

- **[moe_combine_recv](#moe_combine_recv)**

- **[moe_combine_finalize](#moe_combine_finalize)**

先创建 KuRMCL 通信器，再初始化 dispatch/combine。每批次先提交 send，再通过 recv 等待并处理结果；使用结束后调用对应 finalize。

###### moe_dispatch_init

初始化 MoE 分发数据结构，完成内存注册和地址交换。

**接口定义**

```cpp
void moe_dispatch_init(uint8_t *x_data, int16_t *recv_src_info_data,
    int16_t *recv_src_info_bak_data, int64_t num_experts, int64_t multiple,
    int64_t num_max_dispatch_tokens_per_rank, int64_t hidden, int64_t max_num_tokens, int64_t dtp, int16_t *src_info,
    void *disbuf_baseptr, kurmcl_conn_info_h ds_conn_info);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|x_data|`uint8_t *`|存放 token 的发送缓冲区。|输入|
|recv_src_info_data|`int16_t *`|记录接收 token 及源进程编号的主表。|输入|
|recv_src_info_bak_data|`int16_t *`|备用表，与主表交替使用以防止覆盖。|输入|
|num_experts|`int64_t`|专家数。|输入|
|multiple|`int64_t`|预填充场景下接收缓冲区大小的倍数系数。|输入|
|num_max_dispatch_tokens_per_rank|`int64_t`|每个进程最多分发的 token 数。|输入|
|hidden|`int64_t`|token 的特征维度。|输入|
|max_num_tokens|`int64_t`|最大批次大小。|输入|
|dtp|`int64_t`|专家并行系数，dtp = ep / dp。|输入|
|src_info|`int16_t *`|发送端的 token 目的地信息表。|输入|
|disbuf_baseptr|`void *`|分发接收缓冲区的基地址。|输入|
|ds_conn_info|`kurmcl_conn_info_h`|KuRMCL 通信器句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::moe_dispatch_init(
    x_data, recv_src_info_data, recv_src_info_bak_data, num_experts, multiple, 
    num_max_dispatch_tokens_per_rank, hidden, max_num_tokens, dtp, src_info, disbuf_baseptr, 
    ds_conn_info);
```

###### moe_dispatch_send

依据 TopK 路由信息将 token 分发到对应专家。

**接口定义**

```cpp
void moe_dispatch_send(uint8_t *x_data, int16_t *topk_idx_data, int64_t num_tokens, int64_t num_topk,
    int64_t num_max_dispatch_tokens_per_rank, int64_t hidden, int16_t *parallel_policy_data,
    int64_t batch_id, kurmcl_conn_info_h ds_conn_info);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|x_data|`uint8_t *`|存放 token 的发送缓冲区。|输入|
|topk_idx_data|`int16_t *`|token 分发的 TopK 路由表。|输入|
|num_tokens|`int64_t`|批次大小。|输入|
|num_topk|`int64_t`|token 路由的 TopK 值。|输入|
|num_max_dispatch_tokens_per_rank|`int64_t`|每个进程最多分发的 token 数。|输入|
|hidden|`int64_t`|token 的特征维度。|输入|
|parallel_policy_data|`int16_t *`|并行策略数组，存放 (ep, dp, tp)。|输入|
|batch_id|`int64_t`|当前分发任务的批次编号。|输入|
|ds_conn_info|`kurmcl_conn_info_h`|KuRMCL 通信器句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::moe_dispatch_send(
    x_data, topk_idx_data, num_tokens, num_topk, num_max_dispatch_tokens_per_rank, hidden, 
    parallel_policy_data, batch_id, ds_conn_info);
```

###### moe_dispatch_recv

阻塞等待指定批次的 token 接收完成。

**接口定义**

```cpp
void moe_dispatch_recv(int64_t batch_id, kurmcl_conn_info_h ds_conn_info);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|batch_id|`int64_t`|当前接收任务的批次编号。|输入|
|ds_conn_info|`kurmcl_conn_info_h`|KuRMCL 通信器句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::moe_dispatch_recv(batch_id, ds_conn_info);
```

###### moe_dispatch_finalize

释放 MoE 分发相关数据结构。

**接口定义**

```cpp
void moe_dispatch_finalize();
```

**参数**

无。

**返回值**

无（`void`）。

**示例**

```cpp
kutacc::moe_dispatch_finalize();
```

###### moe_combine_init

初始化 MoE 合并数据结构，完成内存注册和地址交换。

**接口定义**

```cpp
void moe_combine_init(bfloat16_t *new_packed_recv_x_data, int64_t num_tokens, int16_t num_experts,
    int16_t num_max_dispatch_tokens_per_rank, int16_t n_activated_experts, int16_t hidden,
    std::vector<bfloat16_t *> &&group_ptr, int local_rank, std::vector<bfloat16_t *> &&recv_group,
    kurmcl_conn_info_h ds_conn_info);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|new_packed_recv_x_data|`bfloat16_t *`|用于合并的已打包发送缓冲区。|输入|
|num_tokens|`int64_t`|批次大小。|输入|
|num_experts|`int16_t`|专家数。|输入|
|num_max_dispatch_tokens_per_rank|`int16_t`|每个进程最多分发的 token 数。|输入|
|n_activated_experts|`int16_t`|激活专家数。|输入|
|hidden|`int16_t`|token 的特征维度。|输入|
|group_ptr|`std::vector<bfloat16_t *> &&`|存放 DP 组最终 AllGather 结果的数组。|输入|
|local_rank|`int`|当前进程在 DP 通信组中的编号。|输入|
|recv_group|`std::vector<bfloat16_t *> &&`|DP 组共享内存接收缓冲区地址数组。|输入|
|ds_conn_info|`kurmcl_conn_info_h`|KuRMCL 通信器句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::moe_combine_init(
    new_packed_recv_x_data, num_tokens, num_experts, num_max_dispatch_tokens_per_rank, 
    n_activated_experts, hidden, std::move(group_ptr), local_rank, std::move(recv_group), 
    ds_conn_info);
```

###### moe_combine_send

回传专家计算结果，支持控制最终合并阶段是否执行 AllGather。

**接口定义**

```cpp
void moe_combine_send(bfloat16_t *x_data, int16_t *src_info_data,
    int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts, int64_t hidden, int16_t *parallel_policy_data,
    int64_t batch_id, kurmcl_conn_info_h ds_conn_info, bfloat16_t *combined_x_data, int16_t *topk_idx_data,
    float *topk_weights_data, int64_t num_tokens, int64_t num_topk, bool enable_allgather = true);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|x_data|`bfloat16_t *`|已弃用且不使用的缓冲区。|输入|
|src_info_data|`int16_t *`|分发阶段记录的 token 目的地信息。|输入|
|num_max_dispatch_tokens_per_rank|`int64_t`|每个进程最多分发的 token 数。|输入|
|num_experts|`int64_t`|专家数。|输入|
|hidden|`int64_t`|token 的特征维度。|输入|
|parallel_policy_data|`int16_t *`|并行策略数组，存放 (ep, dp, tp)。|输入|
|batch_id|`int64_t`|当前合并任务的批次编号。|输入|
|ds_conn_info|`kurmcl_conn_info_h`|KuRMCL 通信器句柄。|输入|
|combined_x_data|`bfloat16_t *`|保存最终合并结果的缓冲区。|输入|
|topk_idx_data|`int16_t *`|token 分发的 TopK 路由表。|输入|
|topk_weights_data|`float *`|TopK 权重表。|输入|
|num_tokens|`int64_t`|批次大小。|输入|
|num_topk|`int64_t`|token 路由的 TopK 值。|输入|
|enable_allgather|`bool`|是否在最终合并阶段执行 AllGather。 默认值：`true`。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::moe_combine_send(
    x_data, src_info_data, num_max_dispatch_tokens_per_rank, num_experts, hidden, 
    parallel_policy_data, batch_id, ds_conn_info, combined_x_data, topk_idx_data, topk_weights_data, 
    num_tokens, num_topk, enable_allgather);
```

###### moe_combine_recv

等待专家结果接收完成，并依据 TopK 权重执行乘加合并。

**接口定义**

```cpp
void moe_combine_recv(bfloat16_t *combined_x_data, int16_t *topk_idx_data, float *topk_weights_data,
    int64_t num_tokens, int64_t num_max_dispatch_tokens_per_rank, int64_t num_topk, int64_t hidden, int64_t batch_id,
    kupl_shm_win_h win, kurmcl_conn_info_h ds_conn_info);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|combined_x_data|`bfloat16_t *`|保存最终合并结果的缓冲区。|输入/输出|
|topk_idx_data|`int16_t *`|token 分发的 TopK 路由表。|输入|
|topk_weights_data|`float *`|TopK 权重表。|输入|
|num_tokens|`int64_t`|批次大小。|输入|
|num_max_dispatch_tokens_per_rank|`int64_t`|每个进程最多分发的 token 数。|输入|
|num_topk|`int64_t`|token 路由的 TopK 值。|输入|
|hidden|`int64_t`|token 的特征维度。|输入|
|batch_id|`int64_t`|当前合并任务的批次编号。|输入|
|win|`kupl_shm_win_h`|共享内存窗口句柄。|输入|
|ds_conn_info|`kurmcl_conn_info_h`|KuRMCL 通信器句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::moe_combine_recv(
    combined_x_data, topk_idx_data, topk_weights_data, num_tokens, num_max_dispatch_tokens_per_rank, 
    num_topk, hidden, batch_id, win, ds_conn_info);
```

###### moe_combine_finalize

释放 MoE 合并相关数据结构。

**接口定义**

```cpp
void moe_combine_finalize();
```

**参数**

无。

**返回值**

无（`void`）。

**示例**

```cpp
kutacc::moe_combine_finalize();
```

##### pp

源码模块：`kunpeng/kutacc/src/pp`。

- **[pp_init](#pp_init)**

- **[pp_put](#pp_put)**

- **[pp_recv](#pp_recv)**

- **[pp_barrier](#pp_barrier)**

###### pp_init

初始化流水线并行的跨通信器激活传输。

**接口定义**

```cpp
void pp_init(uint8_t *base_ptr, int64_t size, kurmcl_conn_info_h ds_conn_info);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|base_ptr|`uint8_t *`|流水线并行激活传输缓冲区的基地址。|输入|
|size|`int64_t`|流水线并行激活传输缓冲区的最大大小。|输入|
|ds_conn_info|`kurmcl_conn_info_h`|KuRMCL 通信器句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::pp_init(base_ptr, size, ds_conn_info);
```

###### pp_put

将激活数据发送至指定目标进程。

**接口定义**

```cpp
void pp_put(int64_t dest_rank, int64_t send_offset, int64_t size, kurmcl_conn_info_h ds_conn_info);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|dest_rank|`int64_t`|接收激活数据的目标进程全局编号。|输入|
|send_offset|`int64_t`|相对于流水线并行激活传输基地址的偏移。|输入|
|size|`int64_t`|流水线并行激活传输的数据大小。|输入|
|ds_conn_info|`kurmcl_conn_info_h`|KuRMCL 通信器句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::pp_put(dest_rank, send_offset, size, ds_conn_info);
```

###### pp_recv

等待指定源进程的激活数据。

**接口定义**

```cpp
void pp_recv(int64_t src_rank, kurmcl_conn_info_h ds_conn_info);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|src_rank|`int64_t`|发送激活数据的源进程全局编号。|输入|
|ds_conn_info|`kurmcl_conn_info_h`|KuRMCL 通信器句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::pp_recv(src_rank, ds_conn_info);
```

###### pp_barrier

与指定远端进程执行点对点同步屏障。

**接口定义**

```cpp
void pp_barrier(int64_t remote_rank, kurmcl_conn_info_h ds_conn_info);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|remote_rank|`int64_t`|点对点同步的远端进程编号。|输入|
|ds_conn_info|`kurmcl_conn_info_h`|KuRMCL 通信器句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::pp_barrier(remote_rank, ds_conn_info);
```

##### comm

源码模块：`kunpeng/kutacc/src/comm`。

- **[shm_alltoall_request_create](#shm_alltoall_request_create)**

- **[shm_alltoall_request_init](#shm_alltoall_request_init)**

- **[shm_alltoall_request_destroy](#shm_alltoall_request_destroy)**

- **[shm_alltoall2D](#shm_alltoall2d)**

- **[shm_allreduce_request_create](#shm_allreduce_request_create)**

- **[shm_allreduce_request_init](#shm_allreduce_request_init)**

- **[shm_allreduce_request_destroy](#shm_allreduce_request_destroy)**

- **[shm_allreduce](#shm_allreduce)**

- **[shm_reduce_scatter_request_create](#shm_reduce_scatter_request_create)**

- **[shm_reduce_scatter_request_init](#shm_reduce_scatter_request_init)**

- **[shm_reduce_scatter_request_destroy](#shm_reduce_scatter_request_destroy)**

- **[shm_reduce_scatter](#shm_reduce_scatter)**

- **[shm_allgather_request_create](#shm_allgather_request_create)**

- **[shm_allgather_request_init](#shm_allgather_request_init)**

- **[shm_allgather_request_destroy](#shm_allgather_request_destroy)**

- **[shm_batch_allgather](#shm_batch_allgather)**

- **[shm_dual_allgather](#shm_dual_allgather)**

本模块为 DeepSeek 共享内存通信接口，通信组大小当前支持 8 或 16。请求按 create → 分配共享内存（如需要）→ init → 通信 → destroy 的顺序使用。`extra_buffers`、`fence_buffers` 和跨进程通信缓冲区必须为各进程可访问的共享内存。

**相关类型**

```cpp
typedef enum shm_datatype {
    SHM_DATATYPE_INT8 = 0,
    SHM_DATATYPE_UINT8,
    SHM_DATATYPE_BFLOAT16
} shm_datatype_t;

typedef struct shm_alltoall_request *shm_alltoall_request_h;
typedef struct shm_allreduce_request *shm_allreduce_request_h;
typedef struct shm_reduce_scatter_request *shm_reduce_scatter_request_h;
typedef struct shm_allgather_request *shm_allgather_request_h;
```

###### shm_alltoall_request_create

创建共享内存 AllToAll 请求，查询附加缓冲区大小。

**接口定义**

```cpp
void shm_alltoall_request_create(int comm_rank, int comm_size, size_t max_num_elements,
    shm_datatype_t datatype, size_t &extra_buffer_size, shm_alltoall_request_h &request);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|comm_rank|`int`|当前进程在通信组中的相对编号。|输入|
|comm_size|`int`|AllToAll 通信组大小，当前仅支持 8 或 16。|输入|
|max_num_elements|`size_t`|最大元素数。|输入|
|datatype|`shm_datatype_t`|通信缓冲区中的元素类型。|输入|
|extra_buffer_size|`size_t &`|AllToAll 所需的附加缓冲区大小。|输出|
|request|`shm_alltoall_request_h &`|AllToAll 请求句柄。|输出|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::shm_alltoall_request_create(
    comm_rank, comm_size, max_num_elements, datatype, extra_buffer_size, request);
```

###### shm_alltoall_request_init

使用共享内存缓冲区和 KuPL 窗口初始化 AllToAll 请求。

**接口定义**

```cpp
void shm_alltoall_request_init(void **extra_buffers, kupl_shm_win_h comm_win,
    kupl_shm_win_h die_win, kupl_shm_win_h socket_win, kupl_shm_win_h node_win,
    const shm_alltoall_request_h &request);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|extra_buffers|`void **`|AllToAll 所需的附加缓冲区，必须使用共享内存。|输入|
|comm_win|`kupl_shm_win_h`|通信组的 KuPL 窗口。|输入|
|die_win|`kupl_shm_win_h`|die 内的 KuPL 窗口。|输入|
|socket_win|`kupl_shm_win_h`|socket 内的 KuPL 窗口。|输入|
|node_win|`kupl_shm_win_h`|节点内的 KuPL 窗口。|输入|
|request|`const shm_alltoall_request_h &`|AllToAll 请求句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::shm_alltoall_request_init(extra_buffers, comm_win, die_win, socket_win, node_win, request);
```

###### shm_alltoall_request_destroy

销毁共享内存 AllToAll 请求。

**接口定义**

```cpp
void shm_alltoall_request_destroy(const shm_alltoall_request_h &request);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|request|`const shm_alltoall_request_h &`|AllToAll 请求句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::shm_alltoall_request_destroy(request);
```

###### shm_alltoall2D

对二维数据执行共享内存 AllToAll，支持指定通信维度及是否执行通信栅栏。

**接口定义**

```cpp
void shm_alltoall2D(void **send_buffers, void *recv_buffers, int64_t height, int64_t width, int64_t dim,
    bool need_comm_fence, const shm_alltoall_request_h &request);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|send_buffers|`void **`|通信组的发送缓冲区数组，必须使用共享内存。|输入|
|recv_buffers|`void *`|当前进程的接收缓冲区。|输入/输出|
|height|`int64_t`|高度维度的大小。|输入|
|width|`int64_t`|宽度维度的大小。|输入|
|dim|`int64_t`|执行通信的维度。|输入|
|need_comm_fence|`bool`|是否在函数内执行通信栅栏。|输入|
|request|`const shm_alltoall_request_h &`|AllToAll 请求句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::shm_alltoall2D(send_buffers, recv_buffers, height, width, dim, need_comm_fence, request);
```

###### shm_allreduce_request_create

创建共享内存 AllReduce 请求，查询附加缓冲区大小。

**接口定义**

```cpp
void shm_allreduce_request_create(int comm_rank, int comm_size, size_t max_num_elements,
    shm_datatype_t datatype, size_t &extra_buffer_size, shm_allreduce_request_h &request);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|comm_rank|`int`|当前进程在通信组中的相对编号。|输入|
|comm_size|`int`|AllReduce 通信组大小，当前仅支持 8 或 16。|输入|
|max_num_elements|`size_t`|最大元素数。|输入|
|datatype|`shm_datatype_t`|通信缓冲区中的元素类型。|输入|
|extra_buffer_size|`size_t &`|AllReduce 所需的附加缓冲区大小。|输出|
|request|`shm_allreduce_request_h &`|AllReduce 请求句柄。|输出|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::shm_allreduce_request_create(
    comm_rank, comm_size, max_num_elements, datatype, extra_buffer_size, request);
```

###### shm_allreduce_request_init

使用共享内存缓冲区和 KuPL 窗口初始化 AllReduce 请求。

**接口定义**

```cpp
void shm_allreduce_request_init(void **extra_buffers, kupl_shm_win_h comm_win, kupl_shm_win_h die_win,
    kupl_shm_win_h socket_win, kupl_shm_win_h node_win, const shm_allreduce_request_h &request);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|extra_buffers|`void **`|AllReduce 所需的附加缓冲区，必须使用共享内存。|输入|
|comm_win|`kupl_shm_win_h`|通信组的 KuPL 窗口。|输入|
|die_win|`kupl_shm_win_h`|die 内的 KuPL 窗口。|输入|
|socket_win|`kupl_shm_win_h`|socket 内的 KuPL 窗口。|输入|
|node_win|`kupl_shm_win_h`|节点内的 KuPL 窗口。|输入|
|request|`const shm_allreduce_request_h &`|AllReduce 请求句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::shm_allreduce_request_init(extra_buffers, comm_win, die_win, socket_win, node_win, request);
```

###### shm_allreduce_request_destroy

销毁共享内存 AllReduce 请求。

**接口定义**

```cpp
void shm_allreduce_request_destroy(const shm_allreduce_request_h &request);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|request|`const shm_allreduce_request_h &`|AllReduce 请求句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::shm_allreduce_request_destroy(request);
```

###### shm_allreduce

对通信组内共享内存缓冲区执行 AllReduce。

**接口定义**

```cpp
void shm_allreduce(void **buffers, size_t num_elements, const shm_allreduce_request_h &request);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|buffers|`void **`|通信组的缓冲区数组，必须使用共享内存。|输入/输出|
|num_elements|`size_t`|元素数。|输入|
|request|`const shm_allreduce_request_h &`|AllReduce 请求句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::shm_allreduce(buffers, num_elements, request);
```

###### shm_reduce_scatter_request_create

创建共享内存 ReduceScatter 请求，查询同步栅栏缓冲区大小。

**接口定义**

```cpp
void shm_reduce_scatter_request_create(int rank, int comm_size, shm_datatype_t datatype,
    size_t &fence_buffer_size, shm_reduce_scatter_request_h &request);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|rank|`int`|当前进程在通信组中的相对编号。|输入|
|comm_size|`int`|ReduceScatter 通信组大小，当前仅支持 8 或 16。|输入|
|datatype|`shm_datatype_t`|通信缓冲区中的元素类型。|输入|
|fence_buffer_size|`size_t &`|ReduceScatter 所需的附加栅栏缓冲区大小。|输出|
|request|`shm_reduce_scatter_request_h &`|ReduceScatter 请求句柄。|输出|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::shm_reduce_scatter_request_create(rank, comm_size, datatype, fence_buffer_size, request);
```

###### shm_reduce_scatter_request_init

使用共享内存栅栏缓冲区和 KuPL 窗口初始化 ReduceScatter 请求。

**接口定义**

```cpp
void shm_reduce_scatter_request_init(void **fence_buffers, kupl_shm_win_h shm_win_intra_die,
    kupl_shm_win_h shm_win_intra_socket, kupl_shm_win_h shm_win_intra_node,
    const shm_reduce_scatter_request_h &request);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|fence_buffers|`void **`|ReduceScatter 所需的栅栏缓冲区，必须使用共享内存。|输入|
|shm_win_intra_die|`kupl_shm_win_h`|die 内的 KuPL 窗口。|输入|
|shm_win_intra_socket|`kupl_shm_win_h`|socket 内的 KuPL 窗口。|输入|
|shm_win_intra_node|`kupl_shm_win_h`|节点内的 KuPL 窗口。|输入|
|request|`const shm_reduce_scatter_request_h &`|ReduceScatter 请求句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::shm_reduce_scatter_request_init(
    fence_buffers, shm_win_intra_die, shm_win_intra_socket, shm_win_intra_node, request);
```

###### shm_reduce_scatter_request_destroy

销毁共享内存 ReduceScatter 请求。

**接口定义**

```cpp
void shm_reduce_scatter_request_destroy(const shm_reduce_scatter_request_h &request);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|request|`const shm_reduce_scatter_request_h &`|待销毁的 ReduceScatter 请求句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::shm_reduce_scatter_request_destroy(request);
```

###### shm_reduce_scatter

对共享内存中的二维数据执行 ReduceScatter。

**接口定义**

```cpp
void shm_reduce_scatter(void **buffers, int64_t height, int64_t width,
    const shm_reduce_scatter_request_h &request);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|buffers|`void **`|DP 组其他进程的缓冲区地址数组。|输入/输出|
|height|`int64_t`|高度维度参数。|输入|
|width|`int64_t`|宽度维度参数。|输入|
|request|`const shm_reduce_scatter_request_h &`|ReduceScatter 请求句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::shm_reduce_scatter(buffers, height, width, request);
```

###### shm_allgather_request_create

创建共享内存 AllGather 请求。

**接口定义**

```cpp
void shm_allgather_request_create(int rank, int comm_size, shm_allgather_request_h &request);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|rank|`int`|当前进程在通信组中的相对编号。|输入|
|comm_size|`int`|AllGather 通信组大小，当前仅支持 8 或 16。|输入|
|request|`shm_allgather_request_h &`|AllGather 请求句柄。|输出|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::shm_allgather_request_create(rank, comm_size, request);
```

###### shm_allgather_request_init

使用 KuPL 窗口初始化 AllGather 请求。

**接口定义**

```cpp
void shm_allgather_request_init(kupl_shm_win_h shm_win_intra_die,
    kupl_shm_win_h shm_win_intra_socket, kupl_shm_win_h shm_win_intra_node,
    const shm_allgather_request_h &request);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|shm_win_intra_die|`kupl_shm_win_h`|die 内的 KuPL 窗口。|输入|
|shm_win_intra_socket|`kupl_shm_win_h`|socket 内的 KuPL 窗口。|输入|
|shm_win_intra_node|`kupl_shm_win_h`|节点内的 KuPL 窗口。|输入|
|request|`const shm_allgather_request_h &`|AllGather 请求句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::shm_allgather_request_init(shm_win_intra_die, shm_win_intra_socket, shm_win_intra_node, request);
```

###### shm_allgather_request_destroy

销毁共享内存 AllGather 请求。

**接口定义**

```cpp
void shm_allgather_request_destroy(const shm_allgather_request_h &request);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|request|`const shm_allgather_request_h &`|AllGather 请求句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::shm_allgather_request_destroy(request);
```

###### shm_batch_allgather

执行批量共享内存 AllGather，支持分层通信和基础版本选择。

**接口定义**

```cpp
void shm_batch_allgather(int64_t batch, void *sendbuf, int64_t sendcount, void *recvbuf,
    int64_t recvcount, shm_datatype_t datatype, uint8_t **remote_sendbuf, uint8_t **remote_recvbuf,
    int64_t buffer_size, const shm_allgather_request_h &request,
    bool is_hierarchical, bool use_base);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|batch|`int64_t`|批次大小。|输入|
|sendbuf|`void *`|AllGather 发送缓冲区。|输入|
|sendcount|`int64_t`|发送数据的总字节数。|输入|
|recvbuf|`void *`|AllGather 接收缓冲区。|输入/输出|
|recvcount|`int64_t`|接收缓冲区的总字节数。|输入|
|datatype|`shm_datatype_t`|通信缓冲区中的元素类型。|输入|
|remote_sendbuf|`uint8_t **`|DP 进程的发送缓冲区地址数组。|输入|
|remote_recvbuf|`uint8_t **`|DP 进程的接收缓冲区地址数组。|输入/输出|
|buffer_size|`int64_t`|每个共享内存缓冲区的大小。|输入|
|request|`const shm_allgather_request_h &`|AllGather 请求句柄。|输入|
|is_hierarchical|`bool`|是否使用分层通信。|输入|
|use_base|`bool`|AllGather 版本选择开关。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::shm_batch_allgather(
    batch, sendbuf, sendcount, recvbuf, recvcount, datatype, remote_sendbuf, remote_recvbuf, 
    buffer_size, request, is_hierarchical, use_base);
```

###### shm_dual_allgather

执行双路共享内存 AllGather，分别指定两组收发缓冲区及通信组屏障回调。

**接口定义**

```cpp
void shm_dual_allgather(void *src0, int64_t src0_size, void *dst0, int64_t dst0_size,
    void *src1, int64_t src1_size, void *dst1, int64_t dst1_size,
    void **remote_buffers0, void **remote_buffers1, shm_datatype_t datatype,
    const std::function<void()> &barrier, const shm_allgather_request_h &request);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|src0|`void *`|第一路 AllGather 的发送缓冲区地址。|输入|
|src0_size|`int64_t`|第一路 AllGather 的发送数据大小。|输入|
|dst0|`void *`|第一路 AllGather 的本地目标缓冲区。|输出|
|dst0_size|`int64_t`|第一路 AllGather 的本地目标数据大小。|输入|
|src1|`void *`|第二路 AllGather 的发送缓冲区地址。|输入|
|src1_size|`int64_t`|第二路 AllGather 的发送数据大小。|输入|
|dst1|`void *`|第二路 AllGather 的本地目标缓冲区。|输出|
|dst1_size|`int64_t`|第二路 AllGather 的本地目标数据大小。|输入|
|remote_buffers0|`void **`|第一路 AllGather 的接收缓冲区数组，包含所有 DP 进程的地址。|输入/输出|
|remote_buffers1|`void **`|第二路 AllGather 的接收缓冲区数组，包含所有 DP 进程的地址。|输入/输出|
|datatype|`shm_datatype_t`|通信缓冲区中的元素类型。|输入|
|barrier|`const std::function<void()> &`|通信组同步屏障回调函数。|输入|
|request|`const shm_allgather_request_h &`|AllGather 请求句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::shm_dual_allgather(
    src0, src0_size, dst0, dst0_size, src1, src1_size, dst1, dst1_size, remote_buffers0, 
    remote_buffers1, datatype, barrier, request);
```

##### mt

源码模块：`kunpeng/kutacc/src/mt`。

- **[get_thread_num](#get_thread_num)**

- **[get_thread_id](#get_thread_id)**

- **[parallel_for](#parallel_for)**

- **[parallel_barrier](#parallel_barrier)**

- **[global_parallel_launch](#global_parallel_launch)**

`parallel_for` 的处理区间为 `[begin, end)`，`grain_size` 必须大于 0；回调接收各任务的起止位置。并行后端由库的构建配置决定。

###### get_thread_num

查询并行后端的线程总数。

**接口定义**

```cpp
int64_t get_thread_num();
```

**参数**

无。

**返回值**

`int64_t`，线程总数。

**示例**

```cpp
auto result = kutacc::get_thread_num();
```

###### get_thread_id

查询当前线程的编号。

**接口定义**

```cpp
int64_t get_thread_id();
```

**参数**

无。

**返回值**

`int64_t`，当前线程编号。

**示例**

```cpp
auto result = kutacc::get_thread_id();
```

###### parallel_for

在指定区间按任务粒度划分工作，并行执行回调函数。

**接口定义**

```cpp
void parallel_for(int64_t begin, int64_t end, int64_t grain_size,
    const std::function<void(int64_t, int64_t)> &func);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|begin|`int64_t`|处理区间的起始位置。|输入|
|end|`int64_t`|处理区间的结束位置，不包含该位置。|输入|
|grain_size|`int64_t`|每个任务的工作粒度，必须大于 0。|输入|
|func|`const std::function<void(int64_t, int64_t)> &`|并行执行的回调函数，参数为任务的起止位置。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

```cpp
std::vector<int64_t> values(1024);
kutacc::parallel_for(0, static_cast<int64_t>(values.size()), 64,
    [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            values[i] = i;
        }
    });
```

###### parallel_barrier

在并行区域执行线程同步屏障。

**接口定义**

```cpp
void parallel_barrier();
```

**参数**

无。

**返回值**

无（`void`）。

**示例**

```cpp
kutacc::parallel_barrier();
```

###### global_parallel_launch

使用 kupl-global 后端启动全局并行回调。

**接口定义**

```cpp
void global_parallel_launch(const std::function<void()> &f);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|f|`const std::function<void()> &`|全局并行执行的回调函数。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::global_parallel_launch(f);
```

##### async

源码模块：`kunpeng/kutacc/src/async`。

- **[async_tg_task_submit](#async_tg_task_submit)**

- **[async_tg_wait](#async_tg_wait)**

- **[async_tg_create](#async_tg_create)**

- **[async_tg_destroy](#async_tg_destroy)**

- **[async_tg_get_thread_num](#async_tg_get_thread_num)**

- **[async_tg_get_thread_id](#async_tg_get_thread_id)**

相关类型：`typedef struct kuasync_exec *async_tg_h;`。先创建线程组并检查返回值，再提交任务；任务参数在任务完成前保持有效，等待完成后销毁线程组。注意创建成功返回 1，与通用状态码不同。

###### async_tg_task_submit

向异步线程组提交任务函数及其参数。

**接口定义**

```cpp
void async_tg_task_submit(async_tg_h executor, void (*routine)(void *), void *args);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|executor|`async_tg_h`|异步线程组句柄。|输入|
|routine|`void (*)(void *)`|待执行任务的函数指针。|输入|
|args|`void *`|传给任务函数的参数。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::async_tg_task_submit(executor, routine, args);
```

###### async_tg_wait

等待异步线程组中所有运行任务结束。

**接口定义**

```cpp
void async_tg_wait(async_tg_h exec);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|exec|`async_tg_h`|异步线程组句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::async_tg_wait(exec);
```

###### async_tg_create

创建异步线程组，并指定工作线程数及 NUMA 节点内的绑核偏移。

**接口定义**

```cpp
int async_tg_create(async_tg_h &executor, int num_threads, int core_offset_in_numa);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|executor|`async_tg_h &`|创建的线程组句柄。|输出|
|num_threads|`int`|工作线程总数。|输入|
|core_offset_in_numa|`int`|NUMA 节点内用于绑核的核心偏移。|输入|

**返回值**

成功返回 1，失败返回 0。

**示例**

```cpp
kutacc::async_tg_h executor = nullptr;
if (kutacc::async_tg_create(executor, 1, 0) == 1) {
    int value = 0;
    kutacc::async_tg_task_submit(executor, [](void *args) {
        *static_cast<int *>(args) = 1;
    }, &value);
    kutacc::async_tg_wait(executor);
    kutacc::async_tg_destroy(executor);
}
```

###### async_tg_destroy

销毁异步线程组并释放资源。

**接口定义**

```cpp
void async_tg_destroy(async_tg_h &executor);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|executor|`async_tg_h &`|待销毁的线程组句柄。|输入/输出|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::async_tg_destroy(executor);
```

###### async_tg_get_thread_num

查询当前异步线程组的线程总数。

**接口定义**

```cpp
int async_tg_get_thread_num();
```

**参数**

无。

**返回值**

`int`，当前线程组的线程总数。

**示例**

```cpp
auto result = kutacc::async_tg_get_thread_num();
```

###### async_tg_get_thread_id

查询当前异步工作线程的编号。

**接口定义**

```cpp
int async_tg_get_thread_id();
```

**参数**

无。

**返回值**

`int`，当前异步工作线程编号。

**示例**

```cpp
auto result = kutacc::async_tg_get_thread_id();
```

##### mul_scalar_add

源码模块：`kunpeng/kutacc/src/mul_scalar_add`。

- **[mul_scalar_add](#mul_scalar_add-1)**

###### mul_scalar_add

执行逐元素标量乘法与可选累加：`o_ptr[i] = (load_output ? o_ptr[i] : 0) + alpha * i_ptr[i]`。

**接口定义**

```cpp
void mul_scalar_add(bfloat16_t *i_ptr, bfloat16_t *o_ptr, int64_t num, float alpha, bool load_output);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|i_ptr|`bfloat16_t *`|BF16 输入张量。|输入|
|o_ptr|`bfloat16_t *`|BF16 输出张量，原地写入。|输入/输出|
|num|`int64_t`|待处理元素数。|输入|
|alpha|`float`|标量乘数。|输入|
|load_output|`bool`|为 true 时先读取原输出值再累加；否则从零开始。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::mul_scalar_add(i_ptr, o_ptr, num, alpha, load_output);
```

##### math

源码模块：`kunpeng/kutacc/src/math`。

- **[scale_softmax](#scale_softmax)**

###### scale_softmax

对 BF16 激活逐行缩放并执行 Softmax，将结果原地写回；每个线程使用独立的浮点临时缓冲区。

**接口定义**

```cpp
void scale_softmax(__bf16 *act, int64_t act_stride, float *scale, float *buf, int64_t buf_stride,
    int64_t height, int64_t width);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|act|`__bf16 *`|BF16 激活张量，形状为 (height, width)。|输入/输出|
|act_stride|`int64_t`|act 的行步长。|输入|
|scale|`float *`|逐行 float 缩放因子，数组长度为 height。|输入|
|buf|`float *`|float 临时缓冲区，每线程至少需要 width 个元素。|输出|
|buf_stride|`int64_t`|buf 中相邻线程缓冲区的步长。|输入|
|height|`int64_t`|行数。|输入|
|width|`int64_t`|每行的列数。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::scale_softmax(act, act_stride, scale, buf, buf_stride, height, width);
```

##### rope

源码模块：`kunpeng/kutacc/src/rope`。

- **[rope](#rope-1)**

###### rope

依据 token 位置和余弦/正弦缓存对 Query、Key 应用旋转位置编码，写入对应输出张量。

**接口定义**

```cpp
void rope(
    const Tensor<int64_t, 1>& position_ids,
    const Tensor<bfloat16_t, 3>& q,
    const Tensor<bfloat16_t, 3>& k,
    const Tensor<bfloat16_t, 3>& q_out,
    const Tensor<bfloat16_t, 3>& k_out,
    const Tensor<bfloat16_t, 2>& cos_sin_cache
);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|position_ids|`const Tensor<int64_t, 1>&`|int64_t token 位置索引一维张量。|输入|
|q|`const Tensor<bfloat16_t, 3>&`|三维 Query 张量，形状为 (tokens, heads, head_size)。|输入|
|k|`const Tensor<bfloat16_t, 3>&`|三维 Key 张量，形状为 (tokens, heads, head_size)。|输入|
|q_out|`const Tensor<bfloat16_t, 3>&`|三维 Query 输出张量，形状与 q 相同。|输出|
|k_out|`const Tensor<bfloat16_t, 3>&`|三维 Key 输出张量，形状与 k 相同。|输出|
|cos_sin_cache|`const Tensor<bfloat16_t, 2>&`|二维余弦/正弦拼接缓存，形状为 (max_seq_len, head_size)。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::rope(position_ids, q, k, q_out, k_out, cos_sin_cache);
```

##### silu_mul

源码模块：`kunpeng/kutacc/src/silu_mul`。

- **[silu_mul_quant](#silu_mul_quant)**

###### silu_mul_quant

将 gate 部分经 SiLU 激活后与 up 部分相乘，再按行量化为 INT8；输入缓冲区的后半部分会被中间结果覆盖。

**接口定义**

```cpp
void silu_mul_quant(int64_t gateupsize, int64_t gateupnumel, int8_t *outdata_ptr,
    bfloat16_t *gateupdata_ptr, float *online_scalesdata_ptr);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|gateupsize|`int64_t`|每行单个分支的列数，总行宽为 2 * gateupsize。|输入|
|gateupnumel|`int64_t`|gate/up 输入的总元素数，即 height * width。|输入|
|outdata_ptr|`int8_t *`|INT8 输出张量，形状为 (height, gateupsize)。|输出|
|gateupdata_ptr|`bfloat16_t *`|BF16 输入张量，形状为 (height, width)，后半部分会被中间结果覆盖。|输入/输出|
|online_scalesdata_ptr|`float *`|逐行 float 缩放因子，数组长度为 height。|输出|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::silu_mul_quant(gateupsize, gateupnumel, outdata_ptr, gateupdata_ptr, online_scalesdata_ptr);
```

#### AlphaFold3 接口

- **[alphafold3/attention](#alphafold3attention)**

- **[alphafold3/transpose](#alphafold3transpose)**

- **[alphafold3/adap_zero_init](#alphafold3adap_zero_init)**

- **[alphafold3/adap_layernorm](#alphafold3adap_layernorm)**

- **[alphafold3/pack_weight](#alphafold3pack_weight)**

- **[alphafold3/transitionblock_pairformer](#alphafold3transitionblock_pairformer)**

- **[alphafold3/trianglemultiplication](#alphafold3trianglemultiplication)**

- **[alphafold3/communication](#alphafold3communication)**

- **[alphafold3/outer_product_mean](#alphafold3outer_product_mean)**

- **[glu](#glu)**

##### alphafold3/attention

源码模块：`kunpeng/kutacc/src/alphafold3/attention`。

- **[msa_attention_sme](#msa_attention_sme)**

- **[msa_attention_linear](#msa_attention_linear)**

- **[msa_attention_out](#msa_attention_out)**

- **[cross_pack_pair_logits](#cross_pack_pair_logits)**

- **[self_pack_pair_logits](#self_pack_pair_logits)**

- **[grid_pack_pair_logits](#grid_pack_pair_logits)**

- **[grid_flash_attn_mask_bias_bf16_hd16](#grid_flash_attn_mask_bias_bf16_hd16)**

- **[self_flash_attn_mask_bias_bf16_hd24](#self_flash_attn_mask_bias_bf16_hd24)**

- **[cross_flash_attn_mask_bias_bf16_hd32](#cross_flash_attn_mask_bias_bf16_hd32)**

- **[grid_flash_attn_mask_bias_bf16_hd32](#grid_flash_attn_mask_bias_bf16_hd32)**

- **[self_flash_attn_mask_bias_bf16_hd48](#self_flash_attn_mask_bias_bf16_hd48)**

- **[self_attention_out](#self_attention_out)**

- **[self_attention_linear_transpose_nosingle](#self_attention_linear_transpose_nosingle)**

- **[self_attention_linear_transpose](#self_attention_linear_transpose)**

- **[cross_attention_out_pack](#cross_attention_out_pack)**

- **[cross_attention_linear_transpose](#cross_attention_linear_transpose)**

- **[mask_b2f](#mask_b2f)**

- **[gridself_layernorm_bias_bf16_pack](#gridself_layernorm_bias_bf16_pack)**

- **[gridself_attention_out](#gridself_attention_out)**

- **[gridself_attention_bias_linear](#gridself_attention_bias_linear)**

- **[gridself_attention_linear_transpose_trans](#gridself_attention_linear_transpose_trans)**

###### msa_attention_sme

执行 MSA 注意力的 SME 核心计算，依据注意力分数对 Value 特征加权求和。

**接口定义**

```cpp
void msa_attention_sme(float* logits, bfloat16_t* v, bfloat16_t* v_avg, int h, int q, int k, int b, int c);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|logits|`float*`|float 注意力分数，形状为 [h, q, k]。|输入|
|v|`bfloat16_t*`|BF16 Value 输入特征，形状为 [b, h, k, c]。|输入|
|v_avg|`bfloat16_t*`|BF16 Value 加权平均输出，形状为 [b, h, q, c]。|输出|
|h|`int`|注意力头数。|输入|
|q|`int`|Query 序列长度。|输入|
|k|`int`|Key 序列长度。|输入|
|b|`int`|批次大小。|输入|
|c|`int`|每个注意力头的隐藏通道数。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::msa_attention_sme(logits, v, v_avg, h, q, k, b, c);
```

###### msa_attention_linear

执行 MSA 注意力线性投影，生成 pair logits、Value 和门控特征。

**接口定义**

```cpp
void msa_attention_linear(bfloat16_t *act, bfloat16_t *pair_act, bfloat16_t *pair_logits, bfloat16_t *v, bfloat16_t *gate, 
    bfloat16_t *pair_logits_w, bfloat16_t *value_w, bfloat16_t *gating_w, int64_t batch, int64_t seq_len, int64_t nchannels, int64_t nheads, 
    int64_t noutput_channels, int64_t output_batches);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|act|`bfloat16_t *`|输入特征，形状为 [batch, seq_len, nchannels]。|输入|
|pair_act|`bfloat16_t *`|成对输入特征，形状为 [output_batches, seq_len, noutput_channels]。|输入|
|pair_logits|`bfloat16_t *`|pair logits 缓冲区，形状为 [output_batches, seq_len, nheads]。|输出|
|v|`bfloat16_t *`|Value 特征，形状为 [batch, seq_len, nchannels]。|输出|
|gate|`bfloat16_t *`|门控特征，形状为 [batch, seq_len, nchannels]。|输出|
|pair_logits_w|`bfloat16_t *`|pair logits 投影权重，形状为 [noutput_channels, nheads]。|输入|
|value_w|`bfloat16_t *`|Value 投影权重，形状为 [nchannels, nchannels]。|输入|
|gating_w|`bfloat16_t *`|门控投影权重，形状为 [nchannels, nchannels]。|输入|
|batch|`int64_t`|批次维度大小。|输入|
|seq_len|`int64_t`|序列 token 数。|输入|
|nchannels|`int64_t`|输入隐藏维度。|输入|
|nheads|`int64_t`|注意力头数。|输入|
|noutput_channels|`int64_t`|每个头的输出隐藏通道数。|输入|
|output_batches|`int64_t`|输出批次分区大小。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::msa_attention_linear(
    act, pair_act, pair_logits, v, gate, pair_logits_w, value_w, gating_w, batch, seq_len, nchannels, 
    nheads, noutput_channels, output_batches);
```

###### msa_attention_out

融合 MSA 注意力输出线性计算和 Sigmoid 门控。

**接口定义**

```cpp
void msa_attention_out(bfloat16_t *v, bfloat16_t *v_avg, bfloat16_t *gate, bfloat16_t *out, int64_t batch, int64_t seq_len, int64_t nchannels);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|v|`bfloat16_t *`|加权 Value 特征，形状为 [batch, seq_len, nchannels]。|输入|
|v_avg|`bfloat16_t *`|Value 加权平均结果，形状为 [batch, seq_len, nchannels]。|输入|
|gate|`bfloat16_t *`|门控激活，形状为 [batch, seq_len, nchannels]。|输入|
|out|`bfloat16_t *`|融合 MSA 最终输出，形状为 [batch, seq_len, nchannels]。|输出|
|batch|`int64_t`|批次大小。|输入|
|seq_len|`int64_t`|序列长度。|输入|
|nchannels|`int64_t`|隐藏通道数。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::msa_attention_out(v, v_avg, gate, out, batch, seq_len, nchannels);
```

###### cross_pack_pair_logits

将交叉注意力的 pair logits 转换为分块浮点布局。

**接口定义**

```cpp
template <typename T>
void cross_pack_pair_logits(T pair_logits_ori, float* pair_logits_r, int nblock_q, int nblock_kv, int N_a, int H, int Nq, int Nkv);
```

**模板参数**

|参数名|描述|
|---|---|
|T|原始 pair logits 缓冲区的类型（包含指针类型）。|

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|pair_logits_ori|`T`|原始 pair bias logits 输入。|输入|
|pair_logits_r|`float*`|打包后的 float logits 缓冲区。|输出|
|nblock_q|`int`|Query 侧块数。|输入|
|nblock_kv|`int`|Key/Value 侧块数。|输入|
|N_a|`int`|辅助序列维度。|输入|
|H|`int`|注意力头数。|输入|
|Nq|`int`|Query token 数。|输入|
|Nkv|`int`|Key/Value token 数。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

模板类型由实参推导，使用库中已实例化的数据类型。

```cpp
kutacc::cross_pack_pair_logits(pair_logits_ori, pair_logits_r, nblock_q, nblock_kv, N_a, H, Nq, Nkv);
```

###### self_pack_pair_logits

将自注意力的 BF16 pair logits 打包为浮点分块布局。

**接口定义**

```cpp
void self_pack_pair_logits(bfloat16_t* pair_logits_ori, float* pair_logits_r, int nblock_q, int nblock_kv, int H, int Nq, int Nkv);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|pair_logits_ori|`bfloat16_t*`|原始 BF16 pair bias logits 输入。|输入|
|pair_logits_r|`float*`|打包后的 float logits 缓冲区。|输出|
|nblock_q|`int`|Query 分块数量。|输入|
|nblock_kv|`int`|Key/Value 分块数量。|输入|
|H|`int`|注意力头数。|输入|
|Nq|`int`|Query token 数。|输入|
|Nkv|`int`|Key/Value token 数。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::self_pack_pair_logits(pair_logits_ori, pair_logits_r, nblock_q, nblock_kv, H, Nq, Nkv);
```

###### grid_pack_pair_logits

将网格自注意力的 BF16 pair logits 打包为浮点分块布局。

**接口定义**

```cpp
void grid_pack_pair_logits(bfloat16_t* pair_logits_ori, float* pair_logits_r, int nblock_q, int nblock_kv, int H, int Nq, int Nkv);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|pair_logits_ori|`bfloat16_t*`|原始 BF16 pair bias logits 输入。|输入|
|pair_logits_r|`float*`|打包后的 float logits 缓冲区。|输出|
|nblock_q|`int`|Query 分块数量。|输入|
|nblock_kv|`int`|Key/Value 分块数量。|输入|
|H|`int`|注意力头数。|输入|
|Nq|`int`|Query token 数。|输入|
|Nkv|`int`|Key/Value token 数。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::grid_pack_pair_logits(pair_logits_ori, pair_logits_r, nblock_q, nblock_kv, H, Nq, Nkv);
```

###### grid_flash_attn_mask_bias_bf16_hd16

执行带掩码和偏置的网格 Flash Attention，注意力头维度为 16。

**接口定义**

```cpp
void grid_flash_attn_mask_bias_bf16_hd16(int B, int H, int N, int D, bfloat16_t* q, bfloat16_t* k, bfloat16_t* v, bfloat16_t* out, 
    float* mask, float* bias_r, float scale, float m_min);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|B|`int`|批次大小。|输入|
|H|`int`|注意力头数。|输入|
|N|`int`|序列 token 数。|输入|
|D|`int`|每个注意力头的隐藏维度。|输入|
|q|`bfloat16_t*`|Query 特征，形状为 [B, H, N, D]。|输入|
|k|`bfloat16_t*`|Key 特征，形状为 [B, H, N, D]。|输入|
|v|`bfloat16_t*`|Value 特征，形状为 [B, H, N, D]。|输入|
|out|`bfloat16_t*`|注意力加权求和输出。|输出|
|mask|`float*`|float 注意力掩码缓冲区。|输入|
|bias_r|`float*`|预处理后的 float pair bias 缓冲区。|输入|
|scale|`float`|注意力分数的缩放因子。|输入|
|m_min|`float`|掩码下限截断阈值。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::grid_flash_attn_mask_bias_bf16_hd16(B, H, N, D, q, k, v, out, mask, bias_r, scale, m_min);
```

###### self_flash_attn_mask_bias_bf16_hd24

执行带 pair bias 的自注意力 Flash Attention，注意力头维度为 24。

**接口定义**

```cpp
void self_flash_attn_mask_bias_bf16_hd24(int B, int H, int Nq, int Nkv, int D, bfloat16_t* q, bfloat16_t* k, bfloat16_t* v, bfloat16_t* out, 
    float* bias, float* pair_logits_r, float scale, float m_min);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|B|`int`|批次大小。|输入|
|H|`int`|注意力头数。|输入|
|Nq|`int`|Query token 数。|输入|
|Nkv|`int`|Key/Value token 数。|输入|
|D|`int`|每个注意力头的隐藏维度。|输入|
|q|`bfloat16_t*`|Query 特征缓冲区。|输入|
|k|`bfloat16_t*`|Key 特征缓冲区。|输入|
|v|`bfloat16_t*`|Value 特征缓冲区。|输入|
|out|`bfloat16_t*`|注意力输出。|输出|
|bias|`float*`|原始 float pair bias 缓冲区。|输入|
|pair_logits_r|`float*`|打包后的 float pair logits 缓冲区。|输入|
|scale|`float`|logits 缩放因子。|输入|
|m_min|`float`|掩码下限截断值。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::self_flash_attn_mask_bias_bf16_hd24(
    B, H, Nq, Nkv, D, q, k, v, out, bias, pair_logits_r, scale, m_min);
```

###### cross_flash_attn_mask_bias_bf16_hd32

执行带 pair bias 的交叉 Flash Attention，注意力头维度为 32。

**接口定义**

```cpp
void cross_flash_attn_mask_bias_bf16_hd32(int B, int N_a, int H, int Nq, int Nkv, int D, bfloat16_t* q, bfloat16_t* k, bfloat16_t* v, bfloat16_t* out, 
    float* bias, float* pair_logits_r, float scale, float m_min);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|B|`int`|批次大小。|输入|
|N_a|`int`|辅助序列维度。|输入|
|H|`int`|注意力头数。|输入|
|Nq|`int`|Query token 数。|输入|
|Nkv|`int`|Key/Value token 数。|输入|
|D|`int`|每个注意力头的隐藏维度。|输入|
|q|`bfloat16_t*`|Query 特征缓冲区。|输入|
|k|`bfloat16_t*`|Key 特征缓冲区。|输入|
|v|`bfloat16_t*`|Value 特征缓冲区。|输入|
|out|`bfloat16_t*`|交叉注意力输出。|输出|
|bias|`float*`|原始 float pair bias 缓冲区。|输入|
|pair_logits_r|`float*`|打包后的 float pair logits 缓冲区。|输入|
|scale|`float`|logits 缩放因子。|输入|
|m_min|`float`|掩码下限截断值。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::cross_flash_attn_mask_bias_bf16_hd32(
    B, N_a, H, Nq, Nkv, D, q, k, v, out, bias, pair_logits_r, scale, m_min);
```

###### grid_flash_attn_mask_bias_bf16_hd32

执行带掩码和偏置的网格 Flash Attention，注意力头维度为 32。

**接口定义**

```cpp
void grid_flash_attn_mask_bias_bf16_hd32(int B, int H, int N, int D, bfloat16_t* q, bfloat16_t* k, bfloat16_t* v, bfloat16_t* out, 
    float* mask, float* bias_r, float scale, float m_min);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|B|`int`|批次大小。|输入|
|H|`int`|注意力头数。|输入|
|N|`int`|序列 token 数。|输入|
|D|`int`|每个注意力头的隐藏维度。|输入|
|q|`bfloat16_t*`|Query 特征。|输入|
|k|`bfloat16_t*`|Key 特征。|输入|
|v|`bfloat16_t*`|Value 特征。|输入|
|out|`bfloat16_t*`|注意力输出。|输出|
|mask|`float*`|float 注意力掩码缓冲区。|输入|
|bias_r|`float*`|预处理后的 float pair bias 缓冲区。|输入|
|scale|`float`|logits 缩放因子。|输入|
|m_min|`float`|掩码下限截断值。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::grid_flash_attn_mask_bias_bf16_hd32(B, H, N, D, q, k, v, out, mask, bias_r, scale, m_min);
```

###### self_flash_attn_mask_bias_bf16_hd48

执行带 pair bias 的自注意力 Flash Attention，注意力头维度为 48。

**接口定义**

```cpp
void self_flash_attn_mask_bias_bf16_hd48(int B, int H, int Nq, int Nkv, int D, bfloat16_t* q, bfloat16_t* k, bfloat16_t* v, bfloat16_t* out, 
    float* bias, float* pair_logits_r, float scale, float m_min);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|B|`int`|批次大小。|输入|
|H|`int`|注意力头数。|输入|
|Nq|`int`|Query token 数。|输入|
|Nkv|`int`|Key/Value token 数。|输入|
|D|`int`|每个注意力头的隐藏维度。|输入|
|q|`bfloat16_t*`|Query 特征缓冲区。|输入|
|k|`bfloat16_t*`|Key 特征缓冲区。|输入|
|v|`bfloat16_t*`|Value 特征缓冲区。|输入|
|out|`bfloat16_t*`|注意力输出缓冲区。|输出|
|bias|`float*`|原始 float pair bias 缓冲区。|输入|
|pair_logits_r|`float*`|打包后的 float pair logits 缓冲区。|输入|
|scale|`float`|logits 缩放因子。|输入|
|m_min|`float`|掩码下限截断阈值。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::self_flash_attn_mask_bias_bf16_hd48(
    B, H, Nq, Nkv, D, q, k, v, out, bias, pair_logits_r, scale, m_min);
```

###### self_attention_out

融合自注意力加权结果与门控，支持打包输出。

**接口定义**

```cpp
void self_attention_out(bfloat16_t* weighted_avg, bfloat16_t* gate, bfloat16_t* pack_wavg, int ndim, int batch, int seq_len, int nchannels, bool dopack);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|weighted_avg|`bfloat16_t*`|注意力加权求和特征。|输入|
|gate|`bfloat16_t*`|门控激活。|输入|
|pack_wavg|`bfloat16_t*`|打包后的融合输出。|输出|
|ndim|`int`|隐藏通道数。|输入|
|batch|`int`|批次大小。|输入|
|seq_len|`int`|序列 token 数。|输入|
|nchannels|`int`|输入隐藏通道数。|输入|
|dopack|`bool`|是否启用权重/特征打包变换。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::self_attention_out(weighted_avg, gate, pack_wavg, ndim, batch, seq_len, nchannels, dopack);
```

###### self_attention_linear_transpose_nosingle

执行不含 single_cond 分支的自注意力 Q/K/V/G 线性投影及转置。

**接口定义**

```cpp
void self_attention_linear_transpose_nosingle(bfloat16_t* x, bfloat16_t* q_proj, float* q_proj_bias, bfloat16_t* k_proj, bfloat16_t* v_proj, bfloat16_t* g_proj,
     bfloat16_t* q, bfloat16_t* k, bfloat16_t* v, bfloat16_t* g, int batch, int seq_len, int nheads, int head_size);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|x|`bfloat16_t*`|输入特征，形状为 [batch, seq_len, nheads * head_size]。|输入|
|q_proj|`bfloat16_t*`|Q 投影权重，形状为 `[nheads * head_size, nheads * head_size]`。|输入|
|q_proj_bias|`float*`|float Q 偏置，形状为 [nheads * head_size]。|输入|
|k_proj|`bfloat16_t*`|K 投影权重，形状为 `[nheads * head_size, nheads * head_size]`。|输入|
|v_proj|`bfloat16_t*`|V 投影权重，形状为 `[nheads * head_size, nheads * head_size]`。|输入|
|g_proj|`bfloat16_t*`|门控投影权重，形状为 `[nheads * head_size, nheads * head_size]`。|输入|
|q|`bfloat16_t*`|转置后的 Q 特征，形状为 [batch, nheads, seq_len, head_size]。|输出|
|k|`bfloat16_t*`|转置后的 K 特征，形状为 [batch, nheads, seq_len, head_size]。|输出|
|v|`bfloat16_t*`|转置后的 V 特征，形状为 [batch, nheads, seq_len, head_size]。|输出|
|g|`bfloat16_t*`|门控特征，形状为 [batch, seq_len, nheads, head_size]。|输出|
|batch|`int`|批次大小。|输入|
|seq_len|`int`|序列 token 数。|输入|
|nheads|`int`|注意力头数。|输入|
|head_size|`int`|每个注意力头的隐藏维度。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::self_attention_linear_transpose_nosingle(
    x, q_proj, q_proj_bias, k_proj, v_proj, g_proj, q, k, v, g, batch, seq_len, nheads, head_size);
```

###### self_attention_linear_transpose

执行自注意力 Q/K/V/G 线性投影及转置。

**接口定义**

```cpp
void self_attention_linear_transpose(bfloat16_t* x, bfloat16_t* q_proj, float* q_proj_bias, bfloat16_t* k_proj, bfloat16_t* v_proj, bfloat16_t* g_proj,
     bfloat16_t* q, bfloat16_t* k, bfloat16_t* v, bfloat16_t* g, int batch, int seq_len, int nheads, int head_size);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|x|`bfloat16_t*`|输入特征，形状为 [batch, seq_len, nheads * head_size]。|输入|
|q_proj|`bfloat16_t*`|Q 投影权重，形状为 `[nheads * head_size, nheads * head_size]`。|输入|
|q_proj_bias|`float*`|float Q 偏置，形状为 [nheads * head_size]。|输入|
|k_proj|`bfloat16_t*`|K 投影权重，形状为 `[nheads * head_size, nheads * head_size]`。|输入|
|v_proj|`bfloat16_t*`|V 投影权重，形状为 `[nheads * head_size, nheads * head_size]`。|输入|
|g_proj|`bfloat16_t*`|门控投影权重，形状为 `[nheads * head_size, nheads * head_size]`。|输入|
|q|`bfloat16_t*`|转置后的 Q 特征，形状为 [batch, nheads, seq_len, head_size]。|输出|
|k|`bfloat16_t*`|转置后的 K 特征，形状为 [batch, nheads, seq_len, head_size]。|输出|
|v|`bfloat16_t*`|转置后的 V 特征，形状为 [batch, nheads, seq_len, head_size]。|输出|
|g|`bfloat16_t*`|门控特征，形状为 [batch, seq_len, nheads, head_size]。|输出|
|batch|`int`|批次大小。|输入|
|seq_len|`int`|序列 token 数。|输入|
|nheads|`int`|注意力头数。|输入|
|head_size|`int`|每个注意力头的隐藏维度。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::self_attention_linear_transpose(
    x, q_proj, q_proj_bias, k_proj, v_proj, g_proj, q, k, v, g, batch, seq_len, nheads, head_size);
```

###### cross_attention_out_pack

融合交叉注意力加权结果和门控，并打包输出。

**接口定义**

```cpp
void cross_attention_out_pack(bfloat16_t* weighted_avg, bfloat16_t* gate, bfloat16_t* pack_wavg, int ndim, int batch, int seq_len, int n_res, int nchannels);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|weighted_avg|`bfloat16_t*`|交叉注意力的 Value 加权和。|输入|
|gate|`bfloat16_t*`|门控激活。|输入|
|pack_wavg|`bfloat16_t*`|打包后的融合输出特征。|输出|
|ndim|`int`|隐藏通道数。|输入|
|batch|`int`|Query 批次大小。|输入|
|seq_len|`int`|Query 序列长度。|输入|
|n_res|`int`|Key/Value 序列长度。|输入|
|nchannels|`int`|输入隐藏通道数。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::cross_attention_out_pack(weighted_avg, gate, pack_wavg, ndim, batch, seq_len, n_res, nchannels);
```

###### cross_attention_linear_transpose

执行交叉注意力 Q/K/V/G 线性投影及转置。

**接口定义**

```cpp
void cross_attention_linear_transpose(bfloat16_t* x_q, bfloat16_t* x_k, bfloat16_t* q_proj, float* q_proj_bias, bfloat16_t* k_proj, 
    bfloat16_t* v_proj, bfloat16_t* g_proj, bfloat16_t* q, bfloat16_t* k, bfloat16_t* v, bfloat16_t* g, int batch_q, int seq_len_q, int batch_k, int seq_len_k, int nheads, int head_size);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|x_q|`bfloat16_t*`|Query 侧输入特征，形状为 [batch_q, seq_len_q, nheads * head_size]。|输入|
|x_k|`bfloat16_t*`|Key/Value 侧输入特征，形状为 [batch_k, seq_len_k, nheads * head_size]。|输入|
|q_proj|`bfloat16_t*`|Q 投影权重，形状为 `[nheads * head_size, nheads * head_size]`。|输入|
|q_proj_bias|`float*`|float Q 偏置，形状为 [nheads * head_size]。|输入|
|k_proj|`bfloat16_t*`|K 投影权重，形状为 `[nheads * head_size, nheads * head_size]`。|输入|
|v_proj|`bfloat16_t*`|V 投影权重，形状为 `[nheads * head_size, nheads * head_size]`。|输入|
|g_proj|`bfloat16_t*`|门控投影权重，形状为 `[nheads * head_size, nheads * head_size]`。|输入|
|q|`bfloat16_t*`|转置后的 Q 特征，形状为 [batch_q, nheads, seq_len_q, head_size]。|输出|
|k|`bfloat16_t*`|转置后的 K 特征，形状为 [batch_k, nheads, seq_len_k, head_size]。|输出|
|v|`bfloat16_t*`|转置后的 V 特征，形状为 [batch_k, nheads, seq_len_k, head_size]。|输出|
|g|`bfloat16_t*`|门控特征，形状为 [batch_q, seq_len_q, nheads, head_size]。|输出|
|batch_q|`int`|Query 批次大小。|输入|
|seq_len_q|`int`|Query token 数。|输入|
|batch_k|`int`|Key/Value 批次大小。|输入|
|seq_len_k|`int`|Key/Value token 数。|输入|
|nheads|`int`|注意力头数。|输入|
|head_size|`int`|每个注意力头的隐藏维度。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::cross_attention_linear_transpose(
    x_q, x_k, q_proj, q_proj_bias, k_proj, v_proj, g_proj, q, k, v, g, batch_q, seq_len_q, batch_k, 
    seq_len_k, nheads, head_size);
```

###### mask_b2f

将布尔注意力掩码转换为浮点掩码。

**接口定义**

```cpp
void mask_b2f(bool* mask, float* mask_data, int B, int N);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|mask|`bool*`|布尔掩码缓冲区，形状为 [B, N]。|输入|
|mask_data|`float*`|转换后的 float 掩码缓冲区。|输出|
|B|`int`|批次大小。|输入|
|N|`int`|序列 token 数。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::mask_b2f(mask, mask_data, B, N);
```

###### gridself_layernorm_bias_bf16_pack

融合网格自注意力 LayerNorm、偏置、BF16 转换和打包。

**接口定义**

```cpp
void gridself_layernorm_bias_bf16_pack(float* x, bfloat16_t* x_bf16, bfloat16_t* x_r, float* ln_weight,  float* ln_bias, int batch, int seq_len, int nchannels, float eps);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|x|`float*`|原始 float 输入特征。|输入|
|x_bf16|`bfloat16_t*`|转换为 BF16 的中间结果。|输出|
|x_r|`bfloat16_t*`|打包后的归一化特征。|输出|
|ln_weight|`float*`|LayerNorm 缩放权重。|输入|
|ln_bias|`float*`|LayerNorm 平移偏置。|输入|
|batch|`int`|批次大小。|输入|
|seq_len|`int`|序列 token 数。|输入|
|nchannels|`int`|隐藏通道数。|输入|
|eps|`float`|方差计算的数值稳定项 epsilon。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::gridself_layernorm_bias_bf16_pack(
    x, x_bf16, x_r, ln_weight, ln_bias, batch, seq_len, nchannels, eps);
```

###### gridself_attention_out

融合网格自注意力加权结果与门控，生成输出。

**接口定义**

```cpp
void gridself_attention_out(bfloat16_t* weights, bfloat16_t* weighted_avg, bfloat16_t* gate,  bfloat16_t* out, int batch, int seq_len, int nchannels);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|weights|`bfloat16_t*`|注意力分数的 Softmax 权重。|输入|
|weighted_avg|`bfloat16_t*`|Value 加权求和特征。|输入|
|gate|`bfloat16_t*`|门控激活。|输入|
|out|`bfloat16_t*`|最终融合输出。|输出|
|batch|`int`|批次大小。|输入|
|seq_len|`int`|序列 token 数。|输入|
|nchannels|`int`|隐藏通道数。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::gridself_attention_out(weights, weighted_avg, gate, out, batch, seq_len, nchannels);
```

###### gridself_attention_bias_linear

执行网格自注意力 pair bias 的线性投影。

**接口定义**

```cpp
void gridself_attention_bias_linear(bfloat16_t* input, bfloat16_t* bias_weight, bfloat16_t* bias, int batch, int seq_len, int nchannels, int nheads);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|input|`bfloat16_t*`|pair bias 输入特征。|输入|
|bias_weight|`bfloat16_t*`|偏置投影权重。|输入|
|bias|`bfloat16_t*`|输出 pair bias logits。|输出|
|batch|`int`|批次大小。|输入|
|seq_len|`int`|序列 token 数。|输入|
|nchannels|`int`|隐藏通道数。|输入|
|nheads|`int`|注意力头数。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::gridself_attention_bias_linear(input, bias_weight, bias, batch, seq_len, nchannels, nheads);
```

###### gridself_attention_linear_transpose_trans

融合网格自注意力 Q/K/V/G/pair bias 线性投影和转置，支持通过标志控制第 0、1 维转置。

**接口定义**

```cpp
void gridself_attention_linear_transpose_trans(bfloat16_t* x, bfloat16_t* pack_x, bfloat16_t* q_proj, bfloat16_t* k_proj, bfloat16_t* v_proj, bfloat16_t* g_proj,
bfloat16_t* pbias_proj, bfloat16_t* q, bfloat16_t* k, bfloat16_t* v, bfloat16_t* g, bfloat16_t* pair_bias, int batch, int seq_len, int nheads, int head_size, bool trans);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|x|`bfloat16_t*`|原始输入特征；trans 为 false 时形状为 [batch, seq_len, nheads, head_size]，为 true 时为 [seq_len, batch, nheads, head_size]。|输入|
|pack_x|`bfloat16_t*`|打包输入缓冲区；trans 为 false 时逻辑形状为 [batch, seq_len, nheads, head_size]，为 true 时为 [seq_len, batch, nheads, head_size]。|输入|
|q_proj|`bfloat16_t*`|Q 投影权重，形状为 `[nheads * head_size, nheads * head_size]`。|输入|
|k_proj|`bfloat16_t*`|K 投影权重，形状为 `[nheads * head_size, nheads * head_size]`。|输入|
|v_proj|`bfloat16_t*`|V 投影权重，形状为 `[nheads * head_size, nheads * head_size]`。|输入|
|g_proj|`bfloat16_t*`|门控投影权重，形状为 `[nheads * head_size, nheads * head_size]`。|输入|
|pbias_proj|`bfloat16_t*`|pair bias 投影权重，形状为 [nheads * head_size, nheads]。|输入|
|q|`bfloat16_t*`|转置后的 Q 特征，形状为 [batch, nheads, seq_len, head_size]。|输出|
|k|`bfloat16_t*`|转置后的 K 特征，形状为 [batch, nheads, seq_len, head_size]。|输出|
|v|`bfloat16_t*`|转置后的 V 特征，形状为 [batch, nheads, seq_len, head_size]。|输出|
|g|`bfloat16_t*`|门控特征，形状为 [batch, seq_len, nheads, head_size]。|输出|
|pair_bias|`bfloat16_t*`|pair bias，形状为 [batch, seq_len, nheads]。|输出|
|batch|`int`|批次大小。|输入|
|seq_len|`int`|序列 token 数。|输入|
|nheads|`int`|注意力头数。|输入|
|head_size|`int`|每个注意力头的隐藏维度。|输入|
|trans|`bool`|是否对输入矩阵和 Q/K/V/G 的第 0、1 维进行转置。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::gridself_attention_linear_transpose_trans(
    x, pack_x, q_proj, k_proj, v_proj, g_proj, pbias_proj, q, k, v, g, pair_bias, batch, seq_len, 
    nheads, head_size, trans);
```

##### alphafold3/transpose

源码模块：`kunpeng/kutacc/src/alphafold3/transpose.cpp`。

- **[mha_transpose_s_h](#mha_transpose_s_h)**

###### mha_transpose_s_h

将 BF16 张量布局由 `[B, N, H, D]` 转换为 `[B, H, N, D]`。

**接口定义**

```cpp
void mha_transpose_s_h(bfloat16_t* in, bfloat16_t* o, int B, int H, int N, int D);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|in|`bfloat16_t*`|输入张量，形状为 [B, N, H, D]。|输入|
|o|`bfloat16_t*`|输出张量，形状为 [B, H, N, D]。|输出|
|B|`int`|批次大小。|输入|
|H|`int`|注意力头维度。|输入|
|N|`int`|token 维度。|输入|
|D|`int`|通道维度，该转置内核要求为 1。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::mha_transpose_s_h(in, o, B, H, N, D);
```

##### alphafold3/adap_zero_init

源码模块：`kunpeng/kutacc/src/alphafold3/adap_zero_init.cpp`。

- **[linear_nobias](#linear_nobias)**

- **[linear_bias_sigmoid](#linear_bias_sigmoid)**

###### linear_nobias

执行无偏置线性投影，支持预打包输入及不同输出数据类型。

**接口定义**

```cpp
template <typename T>
void linear_nobias(bfloat16_t* xptr, bfloat16_t* wptr, T* resultptr, int bucket, int num_channels, int hidden_size, bool x_packed);
```

**模板参数**

|参数名|描述|
|---|---|
|T|输出元素类型，当前支持 float 和 bfloat16_t。|

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|xptr|`bfloat16_t*`|输入特征缓冲区，形状为 [bucket, hidden_size]。|输入|
|wptr|`bfloat16_t*`|线性权重矩阵，逻辑形状为 [hidden_size, num_channels]。|输入|
|resultptr|`T*`|线性输出缓冲区，形状为 [bucket, num_channels]。|输出|
|bucket|`int`|bucket 分块大小。|输入|
|num_channels|`int`|输出通道数。|输入|
|hidden_size|`int`|输入隐藏维度。|输入|
|x_packed|`bool`|输入矩阵是否已预打包。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

模板类型由实参推导，使用库中已实例化的数据类型。

```cpp
kutacc::linear_nobias(xptr, wptr, resultptr, bucket, num_channels, hidden_size, x_packed);
```

###### linear_bias_sigmoid

融合线性投影、偏置和 Sigmoid，生成浮点中间结果及 BF16 输出。

**接口定义**

```cpp
void linear_bias_sigmoid(bfloat16_t* single_cond_ptr, bfloat16_t* wptr, float* bptr, float* resultptr, bfloat16_t* resultptr_bf16,
    int samples, int bucket, int num_channels, int hidden_size);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|single_cond_ptr|`bfloat16_t*`|输入特征，形状为 [bucket, hidden_size]。|输入|
|wptr|`bfloat16_t*`|线性权重矩阵，逻辑形状为 [hidden_size, num_channels]。|输入|
|bptr|`float*`|float 偏置向量，形状为 [num_channels]。|输入|
|resultptr|`float*`|浮点中间结果缓冲区。|输出|
|resultptr_bf16|`bfloat16_t*`|转换后的 BF16 最终输出，形状为 [samples, bucket, num_channels]。|输出|
|samples|`int`|样本总数。|输入|
|bucket|`int`|bucket 分块大小。|输入|
|num_channels|`int`|输出通道数。|输入|
|hidden_size|`int`|输入隐藏维度。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::linear_bias_sigmoid(
    single_cond_ptr, wptr, bptr, resultptr, resultptr_bf16, samples, bucket, num_channels, 
    hidden_size);
```

##### alphafold3/adap_layernorm

源码模块：`kunpeng/kutacc/src/alphafold3/adap_layernorm.cpp`。

- **[layernorm_noparams](#layernorm_noparams)**

- **[layernorm_pack_bf16](#layernorm_pack_bf16)**

- **[adpln_merge_bf16](#adpln_merge_bf16)**

###### layernorm_noparams

对浮点输入原地执行不带仿射权重和偏置的 LayerNorm。

**接口定义**

```cpp
void layernorm_noparams(float* x, int B, int M, int N, float eps);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|x|`float*`|float 特征张量，形状为 [B, M, N]。|输入/输出|
|B|`int`|批次大小。|输入|
|M|`int`|序列长度。|输入|
|N|`int`|隐藏通道数。|输入|
|eps|`float`|方差计算的数值稳定项 epsilon。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::layernorm_noparams(x, B, M, N, eps);
```

###### layernorm_pack_bf16

融合 LayerNorm、BF16 类型转换和打包。

**接口定义**

```cpp
void layernorm_pack_bf16(float* x, bfloat16_t* x_r, float* ln_weight, float* ln_bias, int M, int N, float eps);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|x|`float*`|float 输入特征，形状为 [M, N]。|输入|
|x_r|`bfloat16_t*`|已打包的 BF16 归一化特征，逻辑形状为 [M, N]。|输出|
|ln_weight|`float*`|LayerNorm 缩放权重。|输入|
|ln_bias|`float*`|LayerNorm 平移偏置。|输入|
|M|`int`|序列长度。|输入|
|N|`int`|隐藏通道数。|输入|
|eps|`float`|方差计算的数值稳定项 epsilon。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::layernorm_pack_bf16(x, x_r, ln_weight, ln_bias, M, N, eps);
```

###### adpln_merge_bf16

执行自适应 LayerNorm，融合条件分支的缩放与偏移，并支持输出打包。

**接口定义**

```cpp
void adpln_merge_bf16(float* x, bfloat16_t* s_c, bfloat16_t* s_w, float* s_b, bfloat16_t* c_b2, int B, int M, int D, int D1, bfloat16_t* pack_x, bool dopack);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|x|`float*`|原始 float 输入特征。|输入|
|s_c|`bfloat16_t*`|BF16 条件缩放偏置。|输入|
|s_w|`bfloat16_t*`|BF16 条件缩放权重。|输入|
|s_b|`float*`|float 条件平移偏置。|输入|
|c_b2|`bfloat16_t*`|额外的条件偏置项。|输入|
|B|`int`|批次大小。|输入|
|M|`int`|序列长度。|输入|
|D|`int`|主分支隐藏维度。|输入|
|D1|`int`|条件分支隐藏维度。|输入|
|pack_x|`bfloat16_t*`|打包后的归一化输出。|输出|
|dopack|`bool`|是否对输出矩阵进行打包。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::adpln_merge_bf16(x, s_c, s_w, s_b, c_b2, B, M, D, D1, pack_x, dopack);
```

##### alphafold3/pack_weight

源码模块：`kunpeng/kutacc/src/alphafold3/pack_weight.cpp`。

- **[bgemm_pack_weight_bf16](#bgemm_pack_weight_bf16)**

- **[glu_pack_weight](#glu_pack_weight)**

###### bgemm_pack_weight_bf16

将 BF16 BGEMM 权重打包为 SME 分块布局。

**接口定义**

```cpp
void bgemm_pack_weight_bf16(bfloat16_t *weight,bfloat16_t *packed_weight,int M, int N);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|weight|`bfloat16_t *`|原始权重矩阵。|输入|
|packed_weight|`bfloat16_t *`|用于 SME BGEMM 的预打包块缓冲区。|输出|
|M|`int`|矩阵第 0 维大小。|输入|
|N|`int`|矩阵第 1 维大小。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::bgemm_pack_weight_bf16(weight, packed_weight, M, N);
```

###### glu_pack_weight

将浮点 GLU 权重打包为模板指定的数据类型，支持带批次维度的权重。

**接口定义**

```cpp
template<typename w_t>
void glu_pack_weight(float *wptr, w_t *packed_weight,int num_channels, int num_intermediate, bool hasbatch);
```

**模板参数**

|参数名|描述|
|---|---|
|w_t|打包权重元素类型，当前支持 float 和 bfloat16_t。|

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|wptr|`float *`|原始 float 权重矩阵。|输入|
|packed_weight|`w_t *`|分块打包的权重输出。|输出|
|num_channels|`int`|输入隐藏通道数。|输入|
|num_intermediate|`int`|GLU 分支中间通道数。|输入|
|hasbatch|`bool`|是否存在外层批次维度。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

模板类型由实参推导，使用库中已实例化的数据类型。

```cpp
kutacc::glu_pack_weight(wptr, packed_weight, num_channels, num_intermediate, hasbatch);
```

##### alphafold3/transitionblock_pairformer

源码模块：`kunpeng/kutacc/src/alphafold3/transitionblock_pairformer.cpp`。

- **[transitionblock_out](#transitionblock_out)**

###### transitionblock_out

执行 Pairformer transition block 的前馈输出融合。

**接口定义**

```cpp
void transitionblock_out(bfloat16_t* act, bfloat16_t* weights, bfloat16_t* out, int batch, int seq_len, int nchannels, int nheads);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|act|`bfloat16_t*`|隐藏激活特征。|输入|
|weights|`bfloat16_t*`|输出投影权重。|输入|
|out|`bfloat16_t*`|transition block 最终输出。|输出|
|batch|`int`|批次大小。|输入|
|seq_len|`int`|序列 token 数。|输入|
|nchannels|`int`|隐藏通道数。|输入|
|nheads|`int`|注意力头数。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::transitionblock_out(act, weights, out, batch, seq_len, nchannels, nheads);
```

##### alphafold3/trianglemultiplication

源码模块：`kunpeng/kutacc/src/alphafold3/trianglemultiplication.cpp`。

- **[pack_kernel_eq1](#pack_kernel_eq1)**

- **[einsum_eq1](#einsum_eq1)**

- **[pack_kernel_eq2](#pack_kernel_eq2)**

- **[einsum_eq2](#einsum_eq2)**

- **[trianglemultiplication](#trianglemultiplication)**

###### pack_kernel_eq1

为三角乘法的 `...cik,...cjk->...cij` 方程打包输入。

**接口定义**

```cpp
void pack_kernel_eq1(bfloat16_t* a, bfloat16_t* packed_a, int B, int N);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|a|`bfloat16_t*`|原始输入矩阵。|输入|
|packed_a|`bfloat16_t*`|打包块缓冲区。|输出|
|B|`int`|批次维度大小。|输入|
|N|`int`|序列内部维度大小。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::pack_kernel_eq1(a, packed_a, B, N);
```

###### einsum_eq1

执行三角乘法的 `...cik,...cjk->...cij` 张量收缩。

**接口定义**

```cpp
void einsum_eq1(bfloat16_t* aptr, bfloat16_t* bptr, bfloat16_t* o, int C, int B, int N);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|aptr|`bfloat16_t*`|已打包的输入 A 矩阵。|输入|
|bptr|`bfloat16_t*`|已打包的输入 B 矩阵。|输入|
|o|`bfloat16_t*`|Einsum 输出特征。|输出|
|C|`int`|通道维度大小。|输入|
|B|`int`|批次维度大小。|输入|
|N|`int`|序列内部维度大小。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::einsum_eq1(aptr, bptr, o, C, B, N);
```

###### pack_kernel_eq2

为三角乘法的 `...ckj,...cki->...cij` 方程打包输入。

**接口定义**

```cpp
void pack_kernel_eq2(bfloat16_t* a, bfloat16_t* packed_a, int N, int B);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|a|`bfloat16_t*`|原始输入矩阵。|输入|
|packed_a|`bfloat16_t*`|eq2 布局的打包块缓冲区。|输出|
|N|`int`|第一内部维度大小。|输入|
|B|`int`|批次维度大小。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::pack_kernel_eq2(a, packed_a, N, B);
```

###### einsum_eq2

执行三角乘法的 `...ckj,...cki->...cij` 张量收缩。

**接口定义**

```cpp
void einsum_eq2(bfloat16_t* aptr, bfloat16_t* bptr, bfloat16_t* o, int C, int B, int N);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|aptr|`bfloat16_t*`|已打包的输入 A 矩阵。|输入|
|bptr|`bfloat16_t*`|已打包的输入 B 矩阵。|输入|
|o|`bfloat16_t*`|Einsum 输出特征。|输出|
|C|`int`|通道维度大小。|输入|
|B|`int`|批次维度大小。|输入|
|N|`int`|序列内部维度大小。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::einsum_eq2(aptr, bptr, o, C, B, N);
```

###### trianglemultiplication

融合三角乘法输出线性投影和 Sigmoid 门控。

**接口定义**

```cpp
void trianglemultiplication(bfloat16_t* input, bfloat16_t* input_act, bfloat16_t* output_w, bfloat16_t* gating_w, bfloat16_t* output, bfloat16_t* gate, 
    int64_t c_0, int64_t c_1, int64_t c_2);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|input|`bfloat16_t*`|原始输入特征。|输入|
|input_act|`bfloat16_t*`|预处理后的激活输入。|输入|
|output_w|`bfloat16_t*`|输出投影权重。|输入|
|gating_w|`bfloat16_t*`|门控投影权重。|输入|
|output|`bfloat16_t*`|三角乘法主输出。|输出|
|gate|`bfloat16_t*`|门控临时缓冲区。|输入|
|c_0|`int64_t`|第一通道分区大小。|输入|
|c_1|`int64_t`|第二通道分区大小。|输入|
|c_2|`int64_t`|第三通道分区大小。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::trianglemultiplication(input, input_act, output_w, gating_w, output, gate, c_0, c_1, c_2);
```

##### alphafold3/communication

源码模块：`kunpeng/kutacc/src/alphafold3/communication`。

- **[shm_all2all](#shm_all2all)**

- **[shm_all_gather_dim1](#shm_all_gather_dim1)**

- **[shm_all_gather](#shm_all_gather)**

本模块的三个模板通信接口当前仅支持 8 个进程。调用前准备好共享内存接收缓冲区、有效的进程编号及 KuPL 共享内存窗口；返回的聚合结果指针指向通信缓冲区。

###### shm_all2all

执行 AlphaFold3 共享内存 AllToAll 数据交换，当前仅支持 8 个进程。

**接口定义**

```cpp
template <typename data_t>
void shm_all2all(const data_t* src, data_t* out_ptr, int64_t m_full, int64_t n_full, int64_t inner_size, 
    int64_t size0, int64_t size1, int64_t input_stride0, int64_t input_stride1, int64_t output_stride0, int64_t output_stride1,
    int64_t m_rank, int64_t m_buffer_size, void * m_kupl_recvbuf, kupl_shm_win_h m_recvbuf_win);
```

**模板参数**

|参数名|描述|
|---|---|
|data_t|输入/通信缓冲区元素类型。|

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|src|`const data_t*`|本地源数据缓冲区。|输入|
|out_ptr|`data_t*`|目标输出缓冲区。|输出|
|m_full|`int64_t`|完整的外层维度大小。|输入|
|n_full|`int64_t`|完整的内部维度大小。|输入|
|inner_size|`int64_t`|单个切片的内部元素数。|输入|
|size0|`int64_t`|本地分区第 0 维大小。|输入|
|size1|`int64_t`|本地分区第 1 维大小。|输入|
|input_stride0|`int64_t`|源张量第 0 维步长。|输入|
|input_stride1|`int64_t`|源张量第 1 维步长。|输入|
|output_stride0|`int64_t`|输出张量第 0 维步长。|输入|
|output_stride1|`int64_t`|输出张量第 1 维步长。|输入|
|m_rank|`int64_t`|当前进程编号。|输入|
|m_buffer_size|`int64_t`|每个进程的接收缓冲区元素数。|输入|
|m_kupl_recvbuf|`void *`|原始共享内存接收缓冲区指针。|输入|
|m_recvbuf_win|`kupl_shm_win_h`|KuPL 共享内存窗口句柄。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

模板类型由实参推导，使用库中已实例化的数据类型。

```cpp
kutacc::shm_all2all(
    src, out_ptr, m_full, n_full, inner_size, size0, size1, input_stride0, input_stride1, 
    output_stride0, output_stride1, m_rank, m_buffer_size, m_kupl_recvbuf, m_recvbuf_win);
```

###### shm_all_gather_dim1

沿第 1 维执行 AlphaFold3 共享内存 AllGather，当前仅支持 8 个进程。

**接口定义**

```cpp
template <typename data_t>
uint8_t* shm_all_gather_dim1(const data_t* src, int64_t B, int64_t H_local, int64_t inner_dims_numel, 
    int64_t m_rank, void * m_kupl_recvbuf, kupl_shm_win_h m_recvbuf_win);
```

**模板参数**

|参数名|描述|
|---|---|
|data_t|输入/通信缓冲区元素类型。|

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|src|`const data_t*`|当前进程的本地输入数据。|输入|
|B|`int64_t`|全局批次维度大小。|输入|
|H_local|`int64_t`|当前进程的本地注意力头分区大小。|输入|
|inner_dims_numel|`int64_t`|内部维度展平后的元素数。|输入|
|m_rank|`int64_t`|当前进程编号。|输入|
|m_kupl_recvbuf|`void *`|共享内存接收缓冲区指针。|输入|
|m_recvbuf_win|`kupl_shm_win_h`|KuPL 共享内存窗口句柄。|输入|

**返回值**

`uint8_t*`，聚合后的完整输出缓冲区指针。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

模板类型由实参推导，使用库中已实例化的数据类型。

```cpp
auto result = kutacc::shm_all_gather_dim1(
    src, B, H_local, inner_dims_numel, m_rank, m_kupl_recvbuf, m_recvbuf_win);
```

###### shm_all_gather

沿第 0 维执行 AlphaFold3 共享内存 AllGather，当前仅支持 8 个进程。

**接口定义**

```cpp
template <typename data_t>
uint8_t* shm_all_gather(const data_t* src, int64_t local_elements, size_t elem_size, 
    int64_t m_rank, int64_t m_world_size, void * m_kupl_recvbuf, kupl_shm_win_h m_recvbuf_win);
```

**模板参数**

|参数名|描述|
|---|---|
|data_t|输入/通信缓冲区元素类型。|

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|src|`const data_t*`|当前进程的本地输入缓冲区。|输入|
|local_elements|`int64_t`|当前进程的本地元素数。|输入|
|elem_size|`size_t`|单个元素的字节数。|输入|
|m_rank|`int64_t`|当前 MPI 进程编号。|输入|
|m_world_size|`int64_t`|全局进程总数。|输入|
|m_kupl_recvbuf|`void *`|原始共享内存接收缓冲区指针。|输入|
|m_recvbuf_win|`kupl_shm_win_h`|KuPL 共享内存窗口句柄。|输入|

**返回值**

`uint8_t*`，拼接聚合后的输出缓冲区指针。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

模板类型由实参推导，使用库中已实例化的数据类型。

```cpp
auto result = kutacc::shm_all_gather(
    src, local_elements, elem_size, m_rank, m_world_size, m_kupl_recvbuf, m_recvbuf_win);
```

##### alphafold3/outer_product_mean

源码模块：`kunpeng/kutacc/src/alphafold3/outer_product_mean.cpp`。

- **[default_block_size](#default_block_size)**

- **[outer_product_mean](#outer_product_mean)**

###### default_block_size

计算 OuterProductMean 分块并行使用的默认左右块大小。

**接口定义**

```cpp
void default_block_size(int64_t n_seq, int64_t n_res_gather, int64_t &left_block_size, int64_t &right_block_size);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|n_seq|`int64_t`|全局序列/批次维度。|输入|
|n_res_gather|`int64_t`|聚合分区的 token 数。|输入|
|left_block_size|`int64_t &`|输出的左侧分块大小。|输出|
|right_block_size|`int64_t &`|输出的右侧分块大小。|输出|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::default_block_size(n_seq, n_res_gather, left_block_size, right_block_size);
```

###### outer_product_mean

执行 BF16 OuterProductMean，融合左右投影、外积、掩码归一化及输出投影。

**接口定义**

```cpp
void outer_product_mean(bfloat16_t* input_act, bfloat16_t* left_proj_w, bfloat16_t* right_proj_w, bfloat16_t* left_proj, bfloat16_t* right_proj, 
    bfloat16_t* left_proj_, bfloat16_t* right_proj_, bfloat16_t* right_proj_new, bfloat16_t* output_w, float* output_b, bfloat16_t* out, bfloat16_t* mask,
    bfloat16_t* norm, int64_t c_i, int64_t c_m, int64_t c_z, int64_t n_res, int64_t n_seq, int64_t n_res_gather, int64_t mask_bias, int64_t mask_stride0, 
    int64_t left_block_size, int64_t right_block_size);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|input_act|`bfloat16_t*`|经 LayerNorm 归一化的输入特征，形状为 [n_seq, n_res, c_m]。|输入|
|left_proj_w|`bfloat16_t*`|经 permute(1, 0) 转置的左投影权重，形状为 [c_i, c_m]。|输入|
|right_proj_w|`bfloat16_t*`|经 permute(1, 0) 转置的右投影权重，形状为 [c_i, c_m]。|输入|
|left_proj|`bfloat16_t*`|左投影临时缓冲区，形状为 [c_i, n_res, n_seq]。|输出|
|right_proj|`bfloat16_t*`|右投影临时缓冲区，形状为 [c_i, n_res, n_seq]。|输出|
|left_proj_|`bfloat16_t*`|变换后的左投影临时缓冲区，形状为 [n_res, c_i, n_seq]。|输出|
|right_proj_|`bfloat16_t*`|变换后的右投影临时缓冲区，形状为 [n_res, c_i, n_seq]。|输出|
|right_proj_new|`bfloat16_t*`|聚合分区的右投影，形状为 [n_res_gather, c_i, n_seq]。|输出|
|output_w|`bfloat16_t*`|展平的输出投影权重，形状为 [c_z, c_i * c_i]。|输入|
|output_b|`float*`|输出投影偏置，形状为 [c_z]。|输入|
|out|`bfloat16_t*`|最终输出，形状为 [n_res, n_res_gather, c_z]。|输出|
|mask|`bfloat16_t*`|转置后的掩码，形状为 [n_res, n_seq]。|输入|
|norm|`bfloat16_t*`|掩码外积的归一化分母，形状为 [n_res, n_res_gather]。|输出|
|c_i|`int64_t`|中间通道数。|输入|
|c_m|`int64_t`|输入隐藏通道数。|输入|
|c_z|`int64_t`|输出隐藏通道数。|输入|
|n_res|`int64_t`|本地序列 token 数。|输入|
|n_seq|`int64_t`|批次大小，原始模型中 B = 1024。|输入|
|n_res_gather|`int64_t`|全局聚合分区的 token 数。|输入|
|mask_bias|`int64_t`|多进程分区的掩码切片偏移。|输入|
|mask_stride0|`int64_t`|掩码张量的行步长。|输入|
|left_block_size|`int64_t`|左侧并行分块大小。|输入|
|right_block_size|`int64_t`|右侧并行分块大小。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::outer_product_mean(
    input_act, left_proj_w, right_proj_w, left_proj, right_proj, left_proj_, right_proj_, 
    right_proj_new, output_w, output_b, out, mask, norm, c_i, c_m, c_z, n_res, n_seq, n_res_gather, 
    mask_bias, mask_stride0, left_block_size, right_block_size);
```

##### glu

源码模块：`kunpeng/kutacc/src/glu`。

- **[glu_pack_weight_bf16](#glu_pack_weight_bf16)**

- **[glu_pack_left_bf16](#glu_pack_left_bf16)**

- **[glu_sigmoid](#glu_sigmoid)**

- **[glu_mask_permute](#glu_mask_permute)**

- **[glu_swish](#glu_swish)**

本模块实现位于 `src/glu`，公开声明位于头文件 AlphaFold3 区域，因此归入 AlphaFold3 接口。`glu_swish` 当前显式实例化的类型组合为 `(bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t)`、`(float, bfloat16_t, float, float)` 和 `(float, float, float, float)`。

###### glu_pack_weight_bf16

打包 GLU 线性权重矩阵，供 BF16 GLU 内核使用。

**接口定义**

```cpp
void glu_pack_weight_bf16(bfloat16_t* weight, bfloat16_t* packed_weight, int M, int N);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|weight|`bfloat16_t*`|原始权重矩阵。|输入|
|packed_weight|`bfloat16_t*`|预打包权重块缓冲区。|输出|
|M|`int`|输出通道数。|输入|
|N|`int`|输入通道数。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::glu_pack_weight_bf16(weight, packed_weight, M, N);
```

###### glu_pack_left_bf16

打包 GLU 左侧输入激活矩阵。

**接口定义**

```cpp
void glu_pack_left_bf16(bfloat16_t* matA, bfloat16_t* packed_matA, int M, int N);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|matA|`bfloat16_t*`|原始输入特征矩阵。|输入|
|packed_matA|`bfloat16_t*`|打包块缓冲区。|输出|
|M|`int`|行维度大小。|输入|
|N|`int`|列维度大小。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::glu_pack_left_bf16(matA, packed_matA, M, N);
```

###### glu_sigmoid

融合 GLU 线性计算与 Sigmoid 激活。

**接口定义**

```cpp
void glu_sigmoid(bfloat16_t* xptr,float* wptr,bfloat16_t* maskptr,bfloat16_t* resultptr, int Bn, int Nb, int num_channels);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|xptr|`bfloat16_t*`|输入特征缓冲区。|输入|
|wptr|`float*`|float 线性权重缓冲区。|输入|
|maskptr|`bfloat16_t*`|BF16 掩码条件缓冲区。|输入|
|resultptr|`bfloat16_t*`|GLU 输出缓冲区。|输出|
|Bn|`int`|批次分块大小。|输入|
|Nb|`int`|token 分块大小。|输入|
|num_channels|`int`|隐藏通道数。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::glu_sigmoid(xptr, wptr, maskptr, resultptr, Bn, Nb, num_channels);
```

###### glu_mask_permute

对 GLU 矩阵乘法结果应用掩码并进行布局置换。

**接口定义**

```cpp
void glu_mask_permute(bfloat16_t* packed_x, bfloat16_t* packed_weight, bfloat16_t* o, float* maskptr, int Bn, int Nb, int C, int C1);
```

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|packed_x|`bfloat16_t*`|已打包的输入特征。|输入|
|packed_weight|`bfloat16_t*`|已打包的线性权重。|输入|
|o|`bfloat16_t*`|布局置换后的输出特征。|输出|
|maskptr|`float*`|掩码条件缓冲区。|输入|
|Bn|`int`|批次块大小。|输入|
|Nb|`int`|token 块大小。|输入|
|C|`int`|主隐藏通道数。|输入|
|C1|`int`|门控分支的隐藏通道数。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

```cpp
kutacc::glu_mask_permute(packed_x, packed_weight, o, maskptr, Bn, Nb, C, C1);
```

###### glu_swish

执行双分支线性投影及 Swish 门控融合，支持模板指定输入、输出和权重数据类型。

**接口定义**

```cpp
template<typename x_t, typename r_t, typename w1_t, typename w2_t>
void glu_swish(x_t* xptr, r_t* resultptr, int shape0, int bucket, int num_channels, int num_intermediate, w1_t* w1, w2_t* w2, bool hasbatch);
```

**模板参数**

|参数名|描述|
|---|---|
|x_t|输入特征元素类型。|
|r_t|输出元素类型。|
|w1_t|第一分支权重类型。|
|w2_t|第二分支权重类型。|

**参数**

|参数名|类型|描述|输入/输出|
|---|---|---|---|
|xptr|`x_t*`|输入特征缓冲区。|输入|
|resultptr|`r_t*`|Swish 融合输出。|输出|
|shape0|`int`|外层批次/token 维度。|输入|
|bucket|`int`|bucket 分块大小。|输入|
|num_channels|`int`|输入隐藏通道数。|输入|
|num_intermediate|`int`|GLU 分支中间通道数。|输入|
|w1|`w1_t*`|第一分支的线性权重。|输入|
|w2|`w2_t*`|第二分支的线性权重。|输入|
|hasbatch|`bool`|是否存在外层批次维度。|输入|

**返回值**

无（`void`）。结果通过输出参数或相关缓冲区返回。

**示例**

以下为调用片段，参数使用上表同名变量；调用前须完成缓冲区分配、数据准备及所属模块要求的初始化。

模板类型由实参推导，使用库中已实例化的数据类型。

```cpp
kutacc::glu_swish(xptr, resultptr, shape0, bucket, num_channels, num_intermediate, w1, w2, hasbatch);
```
