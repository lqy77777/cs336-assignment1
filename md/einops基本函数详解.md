# einops 基本函数详解

> 对应环境：本项目 `.venv` 中的 `einops 0.8.2`  
> 整理日期：2026-09-14  
> 目标：掌握 `rearrange`、`reduce`、`repeat`、`einsum`、`pack`、`unpack`、`parse_shape`，并能在 PyTorch 模型中读写常见的张量变换。

## 1. einops 是什么

`einops` 是一个用“命名轴表达式”描述张量操作的库。它支持 PyTorch、NumPy、JAX、TensorFlow 等多种张量后端。

传统代码经常通过维度编号描述操作：

```python
x = x.permute(0, 2, 1, 3).reshape(batch, seq_len, num_heads * head_dim)
```

einops 则直接写出变换前后的轴：

```python
from einops import rearrange

x = rearrange(x, "batch head seq head_dim -> batch seq (head head_dim)")
```

这两种写法可以完成相同的工作，但 einops 表达式同时记录了每个维度的语义。

三个最核心的函数是：

| 函数 | 元素数量 | 主要用途 |
| --- | ---: | --- |
| `rearrange` | 不变 | 转置、变形、合并/拆分轴、增删长度为 1 的轴 |
| `reduce` | 通常减少 | 求和、平均、最大值、池化等归约操作 |
| `repeat` | 增加或不变 | 复制数据、新增并扩展轴、平铺张量 |

后来加入的常用函数包括：

| 函数 | 主要用途 |
| --- | --- |
| `einsum` | 使用命名轴完成点积、矩阵乘法、批量矩阵乘法等乘积与求和 |
| `pack` / `unpack` | 可逆地合并、恢复形状不同的多个张量 |
| `parse_shape` | 将张量形状解析为“轴名 → 轴长度”的字典 |

## 2. 先读懂 pattern（模式字符串）

多数 einops 函数都使用下面的模式：

```text
输入轴 -> 输出轴
```

例如：

```text
batch height width channel -> batch channel height width
```

假设输入形状为 `(2, 32, 48, 3)`，那么每个名字对应：

| 轴名 | 含义 | 长度 |
| --- | --- | ---: |
| `batch` | 样本数量 | 2 |
| `height` | 图像高度 | 32 |
| `width` | 图像宽度 | 48 |
| `channel` | 通道数 | 3 |

输出端把 `channel` 移到前面，因此输出形状为 `(2, 3, 32, 48)`。

### 2.1 轴名不是变量

表达式中的 `batch`、`height`、`width` 只是轴的名字，不要求 Python 中存在同名变量：

```python
y = rearrange(x, "batch height width channel -> batch channel height width")
```

短名字和长名字都可以：

```python
rearrange(x, "b h w c -> b c h w")
```

建议在复杂代码中使用能表达含义的名字，如 `batch`、`seq`、`head`、`head_dim`。

### 2.2 括号表示轴的组合或拆分

输出端的括号表示合并轴：

```text
batch height width channel -> batch (height width channel)
```

输入端的括号表示将一个轴拆开：

```text
batch (head head_dim) -> batch head head_dim
```

拆分通常至少要告诉 einops 一个因子的长度：

```python
y = rearrange(x, "batch (head head_dim) -> batch head head_dim", head=8)
```

若输入第二维长度为 512，给定 `head=8` 后，einops 会推断 `head_dim=64`。

### 2.3 括号内部的顺序很重要

einops 默认遵循 C-order：括号中右侧的轴变化更快。

```python
rearrange(x, "a b -> (a b)")
rearrange(x, "a b -> (b a)")
```

这两个结果形状相同，但元素排列顺序通常不同。不能因为 shape 相同就认为语义相同。

### 2.4 `1` 与 `()` 表示长度为 1 的轴

```python
y = rearrange(x, "batch channel -> batch channel 1 1")
```

若 `x.shape == (2, 3)`，则 `y.shape == (2, 3, 1, 1)`。

`()`也可以表示长度为 1 的轴：

```python
y = rearrange(x, "batch channel -> batch channel () ()")
```

只能用 `rearrange` 删除长度确实为 1 的轴：

```python
x = torch.zeros(2, 3, 1)
y = rearrange(x, "batch channel 1 -> batch channel")
```

### 2.5 `...` 表示若干未明确命名的轴

```python
y = rearrange(x, "batch ... channel -> batch channel ...")
```

它适合中间轴数量不固定的情况。`...` 代表的整组轴会被整体保留。

## 3. `rearrange`：重新排列张量

### 3.1 接口

```python
rearrange(tensor, pattern, **axes_lengths)
```

参数与返回值：

| 名称 | 含义 |
| --- | --- |
| `tensor` | 输入张量；也可以是形状相同的张量列表 |
| `pattern` | 描述输入轴与输出轴关系的字符串 |
| `axes_lengths` | 可选的轴长度，用于拆分轴或创建可推断的结构 |
| 返回值 | 与输入使用同一后端的张量；若后端允许，可能是输入的 view |

核心性质：`rearrange` 不改变元素总数。

### 3.2 转置轴

```python
import torch
from einops import rearrange

x = torch.zeros(2, 3, 4)  # batch=2, seq=3, dim=4
y = rearrange(x, "batch seq dim -> batch dim seq")

assert y.shape == (2, 4, 3)
```

等价思路：

```python
y = x.permute(0, 2, 1)
```

### 3.3 合并轴（flatten）

```python
x = torch.zeros(2, 3, 4, 5)
y = rearrange(x, "batch channel height width -> batch (channel height width)")

assert y.shape == (2, 60)
```

计算过程：`60 = 3 × 4 × 5`。

### 3.4 拆分轴

```python
x = torch.zeros(2, 12, 5)
y = rearrange(x, "batch (head head_dim) seq -> batch head seq head_dim", head=3)

assert y.shape == (2, 3, 5, 4)
```

einops 根据 `12 = head × head_dim` 和 `head=3` 推断出 `head_dim=4`。

如果 12 不能被给定的 `head` 整除，einops 会报错，而不会静默产生错误形状。

### 3.5 同时拆分、转置和合并

Vision Transformer 常把图像切成 patch：

```python
x = torch.zeros(2, 3, 32, 32)
patches = rearrange(
    x,
    "batch channel (grid_h patch_h) (grid_w patch_w) "
    "-> batch (grid_h grid_w) (channel patch_h patch_w)",
    patch_h=4,
    patch_w=4,
)

assert patches.shape == (2, 64, 48)
```

形状推演：

```text
输入：(2, 3, 32, 32)
高度：32 = 8 × 4
宽度：32 = 8 × 4
patch 数量：8 × 8 = 64
每个 patch 的元素：3 × 4 × 4 = 48
输出：(2, 64, 48)
```

### 3.6 多头注意力中的拆分与合并

拆分 hidden dimension：

```python
x = torch.zeros(2, 10, 512)
heads = rearrange(
    x,
    "batch seq (head head_dim) -> batch head seq head_dim",
    head=8,
)

assert heads.shape == (2, 8, 10, 64)
```

合并回去：

```python
restored = rearrange(
    heads,
    "batch head seq head_dim -> batch seq (head head_dim)",
)

assert restored.shape == (2, 10, 512)
assert torch.equal(restored, x)
```

这两个 pattern 互为逆操作。

### 3.7 新增或删除长度为 1 的轴

```python
x = torch.zeros(2, 3)
y = rearrange(x, "batch channel -> batch channel 1 1")
z = rearrange(y, "batch channel 1 1 -> batch channel")

assert y.shape == (2, 3, 1, 1)
assert z.shape == (2, 3)
```

### 3.8 接收张量列表：stack 或 concatenate

```python
images = [torch.zeros(3, 16, 16) for _ in range(4)]

# 新列表轴被命名为 batch，相当于 stack
batch = rearrange(images, "batch channel height width -> batch channel height width")
assert batch.shape == (4, 3, 16, 16)

# 把列表轴和 width 合并，相当于沿宽度方向拼接
wide = rearrange(images, "batch channel height width -> channel height (batch width)")
assert wide.shape == (3, 16, 64)
```

列表中的张量必须具有相同类型和形状。

## 4. `reduce`：归约一个或多个轴

### 4.1 接口

```python
reduce(tensor, pattern, reduction, **axes_lengths)
```

| 名称 | 含义 |
| --- | --- |
| `tensor` | 输入张量或形状相同的张量列表 |
| `pattern` | 输入轴与输出轴的关系 |
| `reduction` | 归约方式，如 `"sum"`、`"mean"`、`"max"` |
| `axes_lengths` | 拆分轴所需的可选长度 |
| 返回值 | 归约后的同后端张量 |

判断归约轴的关键规则：**出现在输入端、但没有出现在输出端的轴会被归约。**

常见归约方式包括：

```text
min、max、sum、mean、prod、any、all
```

也可以在后端支持的前提下传入自定义归约 callable。

### 4.2 对一个轴求平均

```python
from einops import reduce

x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
y = reduce(x, "batch seq dim -> batch dim", "mean")

assert y.shape == (2, 4)
```

`seq` 没出现在输出端，所以对 `seq` 轴求平均。它对应：

```python
y = x.mean(dim=1)
```

### 4.3 同时归约多个轴

```python
x = torch.ones(2, 3, 4)
y = reduce(x, "batch seq dim -> batch", "sum")

assert y.shape == (2,)
assert torch.equal(y, torch.tensor([12.0, 12.0]))
```

这里 `seq` 和 `dim` 都从输出端消失，因此对两个轴一起求和。

### 4.4 保留长度为 1 的轴

```python
x = torch.zeros(2, 3, 4, 5)
y = reduce(x, "batch channel height width -> batch channel 1 1", "mean")

assert y.shape == (2, 3, 1, 1)
```

这类似于：

```python
y = x.mean(dim=(2, 3), keepdim=True)
```

保留单例轴便于后续广播，例如进行通道归一化。

### 4.5 用轴拆分实现池化

```python
x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
y = reduce(
    x,
    "batch channel (height pool_h) (width pool_w) -> batch channel height width",
    "mean",
    pool_h=2,
    pool_w=2,
)

assert y.shape == (1, 1, 2, 2)
```

执行过程：

```text
4×4 图像
  ↓ 将 height 拆成 height×pool_h
  ↓ 将 width 拆成 width×pool_w
  ↓ pool_h、pool_w 不出现在输出端，因此被求平均
2×2 结果
```

### 4.6 全局平均池化

```python
x = torch.zeros(8, 64, 7, 7)
y = reduce(x, "batch channel height width -> batch channel", "mean")

assert y.shape == (8, 64)
```

## 5. `repeat`：复制或扩展张量

### 5.1 接口

```python
repeat(tensor, pattern, **axes_lengths)
```

| 名称 | 含义 |
| --- | --- |
| `tensor` | 输入张量或形状相同的张量列表 |
| `pattern` | 输入轴与输出轴的关系 |
| `axes_lengths` | 新轴长度或重复次数 |
| 返回值 | 重复后的同后端张量 |

与 `rearrange` 不同，`repeat` 允许输出端出现输入端没有的新轴，但必须给出这个轴的长度。

### 5.2 新增并扩展一个轴

```python
from einops import repeat

x = torch.tensor([10, 20, 30])
y = repeat(x, "channel -> batch channel", batch=2)

assert y.shape == (2, 3)
assert torch.equal(y, torch.tensor([[10, 20, 30], [10, 20, 30]]))
```

### 5.3 将灰度图复制为三通道图

```python
x = torch.zeros(32, 32)
y = repeat(x, "height width -> channel height width", channel=3)

assert y.shape == (3, 32, 32)
```

### 5.4 重复整个轴与重复每个元素

```python
x = torch.tensor([1, 2])

a = repeat(x, "width -> repeat width", repeat=3)
b = repeat(x, "width -> (width repeat)", repeat=3)
c = repeat(x, "width -> (repeat width)", repeat=3)
```

结果分别是：

```text
a = [[1, 2],
     [1, 2],
     [1, 2]]

b = [1, 1, 1, 2, 2, 2]
c = [1, 2, 1, 2, 1, 2]
```

`(width repeat)` 与 `(repeat width)` 的长度都是 6，但元素顺序不同。

### 5.5 为 attention mask 添加 batch 和 head 轴

```python
mask = torch.ones(10, 10, dtype=torch.bool)
batched_mask = repeat(
    mask,
    "query key -> batch head query key",
    batch=2,
    head=8,
)

assert batched_mask.shape == (2, 8, 10, 10)
```

注意：是否真正复制内存取决于后端和具体操作。若张量很大，仍要关注输出的逻辑尺寸和后续算子的内存消耗。

## 6. `einsum`：命名轴的张量乘积与求和

### 6.1 接口

```python
einsum(tensor1, tensor2, ..., pattern)
```

| 名称 | 含义 |
| --- | --- |
| `tensor1, tensor2, ...` | 一个或多个输入张量 |
| `pattern` | 最后一个位置参数；逗号分隔各输入，`->` 后描述输出轴 |
| 返回值 | 乘积并按规则求和后的张量 |

einops 与 `torch.einsum` / `numpy.einsum` 的重要区别：

- einops 使用可读的多字符轴名，如 `batch`、`query`、`key`。
- pattern 放在所有输入张量之后。
- 没有出现在输出端的轴会被求和。
- `einsum` 当前不支持 `rearrange` 式的复合轴括号，也不支持用 `()` 创建单例轴。

### 6.2 向量点积

```python
from einops import einsum

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
result = einsum(x, y, "dim, dim ->")

assert result.item() == 32.0
```

计算为：

```text
1×4 + 2×5 + 3×6 = 32
```

`dim` 没出现在输出端，所以沿 `dim` 求和。

### 6.3 矩阵乘法

```python
a = torch.zeros(2, 3)
b = torch.zeros(3, 4)
c = einsum(a, b, "row inner, inner col -> row col")

assert c.shape == (2, 4)
```

`inner` 是两个输入共享、但输出中不存在的轴，所以它是矩阵乘法的求和轴。

### 6.4 批量矩阵乘法

```python
a = torch.zeros(8, 10, 64)
b = torch.zeros(8, 64, 20)
c = einsum(
    a,
    b,
    "batch row inner, batch inner col -> batch row col",
)

assert c.shape == (8, 10, 20)
```

`batch` 出现在输出端，因此不同 batch 之间不会混合。

### 6.5 attention score

```python
query = torch.zeros(2, 8, 10, 64)
key = torch.zeros(2, 8, 12, 64)

scores = einsum(
    query,
    key,
    "batch head query_seq head_dim, "
    "batch head key_seq head_dim "
    "-> batch head query_seq key_seq",
)

assert scores.shape == (2, 8, 10, 12)
```

`head_dim` 没有出现在输出端，因此 query 和 key 沿 `head_dim` 做点积。`query_seq` 与 `key_seq` 都被保留，构成注意力矩阵的两个维度。

### 6.6 attention value aggregation

```python
attention = torch.zeros(2, 8, 10, 12)
value = torch.zeros(2, 8, 12, 64)

output = einsum(
    attention,
    value,
    "batch head query_seq key_seq, "
    "batch head key_seq head_dim "
    "-> batch head query_seq head_dim",
)

assert output.shape == (2, 8, 10, 64)
```

这里 `key_seq` 是求和轴。

## 7. `pack` 与 `unpack`：可逆地打包多个张量

### 7.1 为什么需要它们

`stack` 要求所有张量形状相同；`concatenate` 要求除拼接轴外的其他轴一致。实际模型中，不同来源的 token 可能具有不同数量的中间轴：

```text
class token:  (batch, 1, dim)
image token:  (batch, height, width, dim)
text token:   (batch, seq, dim)
```

`pack` 可以把通配符 `*` 覆盖的轴压成一个公共轴，同时记录恢复形状所需的信息；`unpack` 再将结果恢复。

### 7.2 `pack` 接口

```python
packed, packed_shapes = pack(tensors, pattern)
```

| 名称 | 含义 |
| --- | --- |
| `tensors` | 待打包的张量序列 |
| `pattern` | 必须包含一个 `*`，表示每个输入中形状可变的部分 |
| `packed` | 打包后的单个张量 |
| `packed_shapes` | 每个输入被 `*` 匹配到的原始形状，供 `unpack` 使用 |

### 7.3 `unpack` 接口

```python
outputs = unpack(tensor, packed_shapes, pattern)
```

| 名称 | 含义 |
| --- | --- |
| `tensor` | 待恢复的已打包张量 |
| `packed_shapes` | `pack` 返回的形状信息 |
| `pattern` | 与打包时相同的模式 |
| 返回值 | 按原顺序恢复得到的张量列表 |

### 7.4 完整例子

```python
from einops import pack, unpack

class_token = torch.zeros(2, 1, 16)
image_tokens = torch.zeros(2, 4, 4, 16)
text_tokens = torch.zeros(2, 5, 16)

packed, packed_shapes = pack(
    [class_token, image_tokens, text_tokens],
    "batch * dim",
)

assert packed.shape == (2, 22, 16)  # 1 + 4×4 + 5 = 22

class_out, image_out, text_out = unpack(
    packed,
    packed_shapes,
    "batch * dim",
)

assert class_out.shape == (2, 1, 16)
assert image_out.shape == (2, 4, 4, 16)
assert text_out.shape == (2, 5, 16)
```

数据流可以理解为：

```text
(b, 1, d) ───────┐
(b, h, w, d) ────┼─ pack("b * d") → (b, 1+h×w+s, d)
(b, s, d) ───────┘                         │
                                           └─ unpack → 恢复三个原形状
```

`packed_shapes` 是恢复结构所需的元数据，应与 `packed` 一起保留。不要自己猜测或重建它，除非你完全确定每段形状。

## 8. `parse_shape`：获取命名轴的长度

### 8.1 接口

```python
parse_shape(x, pattern)
```

| 名称 | 含义 |
| --- | --- |
| `x` | 任意受支持后端的张量 |
| `pattern` | 用空格分隔的轴名；`_` 表示忽略该轴 |
| 返回值 | 字典，键为轴名，值为相应轴的长度 |

### 8.2 基本例子

```python
from einops import parse_shape

x = torch.zeros(2, 3, 32, 48)
shape = parse_shape(x, "batch channel height width")

assert shape == {
    "batch": 2,
    "channel": 3,
    "height": 32,
    "width": 48,
}
```

忽略不需要的轴：

```python
shape = parse_shape(x, "batch _ height width")
assert shape == {"batch": 2, "height": 32, "width": 48}
```

### 8.3 把解析结果传给其他函数

```python
x = torch.zeros(2, 3, 4)
flat = torch.zeros(24)

restored = rearrange(
    flat,
    "(batch seq dim) -> batch seq dim",
    **parse_shape(x, "batch seq dim"),
)

assert restored.shape == (2, 3, 4)
```

`parse_shape` 不用于解析括号形式的复合轴。对于符号形状后端，返回值也可能是符号而非普通 Python 整数。

## 9. 在 PyTorch 模型中使用 Layer 版本

普通函数适合写在 `forward` 中：

```python
def forward(self, x):
    return rearrange(x, "batch channel height width -> batch (channel height width)")
```

如果希望直接放入 `torch.nn.Sequential`，可以使用 `einops.layers.torch`：

```python
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3),
    Rearrange("batch channel height width -> batch (channel height width)"),
    nn.LazyLinear(10),
)
```

`Rearrange` 的参数和返回结果：

| 名称 | 含义 |
| --- | --- |
| `pattern` | 与函数版 `rearrange` 相同的模式 |
| `axes_lengths` | 可选轴长度 |
| 构造结果 | 一个 PyTorch `nn.Module` |
| forward 返回值 | 重新排列后的张量 |

`Reduce` 示例：

```python
pool = Reduce(
    "batch channel height width -> batch channel",
    "mean",
)

x = torch.zeros(8, 64, 7, 7)
y = pool(x)
assert y.shape == (8, 64)
```

函数版适合一次性操作；Layer 版适合注册到模型结构、保存模型配置，或放入 `Sequential`。

## 10. 与常见 PyTorch 操作的对应关系

| 目标 | PyTorch 思路 | einops 写法 |
| --- | --- | --- |
| 转置 | `x.permute(0, 2, 1)` | `rearrange(x, "b s d -> b d s")` |
| flatten | `x.reshape(b, -1)` | `rearrange(x, "b c h w -> b (c h w)")` |
| 拆分 hidden | `reshape` 后 `permute` | `rearrange(x, "b s (h d) -> b h s d", h=h)` |
| squeeze | `x.squeeze(-1)` | `rearrange(x, "b d 1 -> b d")` |
| unsqueeze | `x.unsqueeze(-1)` | `rearrange(x, "b d -> b d 1")` |
| 按轴求平均 | `x.mean(dim=1)` | `reduce(x, "b s d -> b d", "mean")` |
| keepdim 平均 | `x.mean((2, 3), keepdim=True)` | `reduce(x, "b c h w -> b c 1 1", "mean")` |
| expand/repeat | `unsqueeze` + `expand/repeat` | `repeat(x, "d -> b d", b=b)` |
| 批量矩阵乘法 | `torch.bmm` / `matmul` | `einsum(a, b, "b i k, b k j -> b i j")` |

einops 的优势不是提供所有后端都没有的新能力，而是把多步 shape 操作写成一个可检查的声明。

## 11. 常见错误及排查方法

### 11.1 输入维度数量和 pattern 不一致

```python
x = torch.zeros(2, 3, 4)
rearrange(x, "batch dim -> batch dim")  # 错误：输入是三维，pattern 只有两个轴
```

排查：先打印 `x.shape`，逐个给实际维度命名。

### 11.2 拆分时不能整除

```python
x = torch.zeros(2, 10)
rearrange(x, "batch (head dim) -> batch head dim", head=3)
```

`10` 不能拆成 `3 × dim`，因此会报 shape mismatch。

排查：检查原轴长度是否等于括号中所有轴长度的乘积。

### 11.3 `rearrange` 中凭空增加长度大于 1 的轴

```python
x = torch.zeros(2, 3)
rearrange(x, "batch dim -> batch copy dim", copy=4)  # 错误
```

新增并复制数据应该使用：

```python
repeat(x, "batch dim -> batch copy dim", copy=4)
```

判断方法：

- 只改变观察方式、元素数不变：`rearrange`。
- 删除轴并聚合数据：`reduce`。
- 增加重复数据：`repeat`。

### 11.4 `reduce` 忘记指定 reduction

```python
# 错误：缺少第三个参数
# reduce(x, "batch seq dim -> batch dim")

# 正确
reduce(x, "batch seq dim -> batch dim", "mean")
```

### 11.5 `repeat` 没给新轴长度

```python
# 错误：不知道 batch 多长
# repeat(x, "dim -> batch dim")

# 正确
repeat(x, "dim -> batch dim", batch=8)
```

### 11.6 `einsum` 的 pattern 放错位置

```python
# torch.einsum 风格：pattern 在前
torch.einsum("b i d, b j d -> b i j", q, k)

# einops.einsum 风格：张量在前，pattern 在最后
einsum(q, k, "batch i dim, batch j dim -> batch i j")
```

### 11.7 只检查 shape，不检查元素顺序

```python
a = rearrange(x, "height width -> (height width)")
b = rearrange(x, "height width -> (width height)")
```

`a.shape` 与 `b.shape` 相同，但内容顺序可能不同。学习时最好使用 `torch.arange` 创建可追踪的小张量，而不是全零张量。

### 11.8 后续 `.view()` 遇到非连续内存

转置后的 PyTorch 张量可能不是 contiguous 的。einops 会尽可能返回 view，但不保证每个结果都具有连续内存布局。

若后续代码必须使用要求连续内存的操作，可以先检查：

```python
print(y.is_contiguous())
```

必要时再调用：

```python
y = y.contiguous()
```

不要无条件到处加 `.contiguous()`；它可能产生真实的数据复制。

## 12. 一套读写 pattern 的方法

遇到复杂表达式时，可以按下面顺序处理：

1. 写出输入 shape，例如 `(2, 10, 512)`。
2. 给每个输入维度命名，例如 `batch seq hidden`。
3. 判断操作属于重排、归约还是复制。
4. 若需拆轴，把原轴写成括号，例如 `(head head_dim)`。
5. 在输出端按目标顺序写出轴名。
6. 检查输入端和输出端各轴的去向。
7. 手算输出 shape 和元素总数。
8. 使用 `torch.arange` 的小张量验证元素顺序。

例如要把 `(batch, seq, hidden) = (2, 10, 512)` 变为八个注意力头：

```text
1. 输入：batch seq hidden
2. hidden = head × head_dim
3. head = 8，所以 head_dim = 64
4. 目标顺序：batch head seq head_dim
5. pattern：batch seq (head head_dim) -> batch head seq head_dim
```

对应代码：

```python
y = rearrange(
    x,
    "batch seq (head head_dim) -> batch head seq head_dim",
    head=8,
)
```

## 13. 综合小例子：多头自注意力中的 shape 流程

下面只演示 shape 变换，不包含完整的投影层、mask 和 softmax 数值逻辑：

```python
import math
import torch
from einops import einsum, rearrange

batch = 2
seq = 10
num_heads = 8
head_dim = 64
hidden = num_heads * head_dim

q = torch.randn(batch, seq, hidden)
k = torch.randn(batch, seq, hidden)
v = torch.randn(batch, seq, hidden)

q = rearrange(q, "b s (h d) -> b h s d", h=num_heads)
k = rearrange(k, "b s (h d) -> b h s d", h=num_heads)
v = rearrange(v, "b s (h d) -> b h s d", h=num_heads)

scores = einsum(q, k, "b h query d, b h key d -> b h query key")
scores = scores / math.sqrt(head_dim)
attention = scores.softmax(dim=-1)

context = einsum(
    attention,
    v,
    "b h query key, b h key d -> b h query d",
)

output = rearrange(context, "b h s d -> b s (h d)")

assert scores.shape == (2, 8, 10, 10)
assert context.shape == (2, 8, 10, 64)
assert output.shape == (2, 10, 512)
```

关键状态变化：

```text
q/k/v: (b, s, h×d)
   ↓ rearrange：拆出 head，并调整顺序
q/k/v: (b, h, s, d)
   ↓ einsum：沿 d 做 query-key 点积
scores: (b, h, query, key)
   ↓ softmax 后与 value 沿 key 求和
context: (b, h, query, d)
   ↓ rearrange：合并 h 和 d
output: (b, query, h×d)
```

## 14. 如何选择函数

可以先问自己：“输入端的轴到输出端发生了什么？”

```text
元素只是换位置、换分组？
    └─ rearrange

某些轴消失，并且要对它们求和/平均/最大值？
    └─ reduce

输出增加了重复的数据或新的非单例轴？
    └─ repeat

多个张量相乘，并沿共享轴求和？
    └─ einsum

多个不同 rank/shape 的张量要临时拼成一个，再精确恢复？
    └─ pack + unpack

只想按名字读取 shape？
    └─ parse_shape
```

## 15. 建议的自查练习

### 练习 1：轴转置

将 `(batch, height, width, channel)` 转成 `(batch, channel, height, width)`。

```python
rearrange(x, "b h w c -> b c h w")
```

### 练习 2：拆分注意力头

将 `(2, 12, 768)` 转成 `(2, 12, 12, 64)`，分别表示 `batch, head, seq, head_dim`。

```python
rearrange(x, "b s (h d) -> b h s d", h=12)
```

### 练习 3：序列平均

将 `(batch, seq, dim)` 沿 `seq` 求平均。

```python
reduce(x, "b s d -> b d", "mean")
```

### 练习 4：复制 batch

将一个 `(seq, dim)` 张量复制成 `(8, seq, dim)`。

```python
repeat(x, "s d -> b s d", b=8)
```

### 练习 5：计算注意力矩阵

`q` 和 `k` 均按 `(batch, head, seq, head_dim)` 排列，得到 `(batch, head, query, key)`。

```python
einsum(q, k, "b h query d, b h key d -> b h query key")
```

## 16. 补充实例：用具体数值理解每一步

本节每个 Python 代码块都可以独立运行。先根据解释预测结果，再运行代码中的断言检查理解。例子采用小张量，便于追踪元素，而不仅仅检查形状。

### 16.1 转置后再展平：为什么括号顺序会改变结果

变量含义：`x` 是两行三列的矩阵，`row`、`col` 分别是行轴和列轴。`row_first` 按原行顺序展平；`col_first` 先交换行列再展平。两者都是长度为 6 的返回张量。

```python
import torch
from einops import rearrange

x = torch.tensor([[0, 1, 2], [3, 4, 5]])
transposed = rearrange(x, "row col -> col row")
row_first = rearrange(x, "row col -> (row col)")
col_first = rearrange(x, "row col -> (col row)")

assert transposed.tolist() == [[0, 3], [1, 4], [2, 5]]
assert row_first.tolist() == [0, 1, 2, 3, 4, 5]
assert col_first.tolist() == [0, 3, 1, 4, 2, 5]
```

第一步建立有明确数值的矩阵；第二步使 `transposed[col, row] = x[row, col]`；最后两步分别按 `(row col)` 与 `(col row)` 遍历。括号中的最后一个轴变化最快，因此第二种展平方式先走完一列。

### 16.2 拆分一维数组：连续分组与交错分组

`x` 包含 0 到 11。`group=3` 表示分成三组，`item` 的长度由 `12 / 3 = 4` 推断。`continuous` 和 `interleaved` 的返回形状都为 `(3, 4)`。

```python
import torch
from einops import rearrange

x = torch.arange(12)
continuous = rearrange(x, "(group item) -> group item", group=3)
interleaved = rearrange(x, "(item group) -> group item", group=3)

assert continuous.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
assert interleaved.tolist() == [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]
```

连续分组中，原下标等于 `group_index * 4 + item_index`；交错分组中，原下标等于 `item_index * 3 + group_index`。这解释了为什么两种写法形状相同、内容分组却不同。

### 16.3 用非零值验证多头拆分的逆操作

`x` 是一个 batch、两个 token、每个 token 六个特征。`head=2`，所以每个头有三个特征。`heads` 返回 `(1, 2, 2, 3)`，四个维度依次为 batch、head、seq、dim。

```python
import torch
from einops import rearrange

x = torch.arange(12).reshape(1, 2, 6)
heads = rearrange(x, "b s (h d) -> b h s d", h=2)
restored = rearrange(heads, "b h s d -> b s (h d)")

assert heads[0, 0].tolist() == [[0, 1, 2], [6, 7, 8]]
assert heads[0, 1].tolist() == [[3, 4, 5], [9, 10, 11]]
assert torch.equal(restored, x)
```

先把每个 token 的六个特征分成两段，再把 head 轴移到 seq 前面。第一个头收集每个 token 的前三个特征；第二个头收集后三个。恢复时必须先回到 token 优先的排列，再合并头与特征。

### 16.4 用 `...` 交换最后两个轴

`x` 的形状为 `(2, 3, 4, 5)`。`...` 匹配前面的 `(2, 3)`，`row=4`、`col=5`。返回张量 `y` 保留前两个轴，交换后两个轴。

```python
import torch
from einops import rearrange

x = torch.arange(120).reshape(2, 3, 4, 5)
y = rearrange(x, "... row col -> ... col row")

assert y.shape == (2, 3, 5, 4)
assert y[1, 2, 4, 3].item() == x[1, 2, 3, 4].item()
assert torch.equal(y, x.transpose(-1, -2))
```

这种写法也能用于二维或三维输入，只要输入至少有两个轴。它适合“前面可能有若干 batch 轴，最后两个轴是矩阵”的情况。

### 16.5 四个 patch 的具体内容与恢复

`image` 是一张 4×4 单通道图。`gh/gw` 表示 patch 网格位置，`ph/pw` 表示 patch 内部位置，二者都取 2。`patches` 返回四个长度为 4 的 patch；`restored` 返回原图。

```python
import torch
from einops import rearrange

image = torch.arange(16).reshape(4, 4)
patches = rearrange(image, "(gh ph) (gw pw) -> (gh gw) (ph pw)", ph=2, pw=2)
restored = rearrange(
    patches, "(gh gw) (ph pw) -> (gh ph) (gw pw)", gh=2, gw=2, ph=2, pw=2
)

assert patches.tolist() == [
    [0, 1, 4, 5],
    [2, 3, 6, 7],
    [8, 9, 12, 13],
    [10, 11, 14, 15],
]
assert torch.equal(restored, image)
```

输入端将行列各拆成“网格位置 × 块内位置”；输出端先组合网格位置，再组合块内位置。因此四个 patch 依次对应左上、右上、左下、右下。

### 16.6 同一矩阵沿不同轴求和

`x` 是两行三列的矩阵。`row_sums` 为每行之和，`col_sums` 为每列之和，`total` 是零维张量。注意“保留哪个轴”与“对哪个轴求和”是两件事。

```python
import torch
from einops import reduce

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
row_sums = reduce(x, "row col -> row", "sum")
col_sums = reduce(x, "row col -> col", "sum")
total = reduce(x, "row col ->", "sum")

assert row_sums.tolist() == [6, 15]
assert col_sums.tolist() == [5, 7, 9]
assert total.shape == torch.Size([])
assert total.item() == 21
```

第一种删除 `col`，所以在每一行内部相加；第二种删除 `row`，所以在每一列内部相加；第三种两个轴都删除，将全部元素归约成一个标量。

### 16.7 保留单例轴，实现逐行中心化

`x` 有两行，每行三个浮点数。`means` 保存每行均值，形状为 `(2, 1)`；`centered` 是原矩阵减去各自的行均值，形状仍为 `(2, 3)`。

```python
import torch
from einops import reduce

x = torch.tensor([[1., 2., 3.], [10., 20., 30.]])
means = reduce(x, "row col -> row 1", "mean")
centered = x - means

assert means.tolist() == [[2.], [20.]]
assert centered.tolist() == [[-1., 0., 1.], [-10., 0., 10.]]
assert torch.allclose(centered.mean(dim=1), torch.zeros(2))
```

归约时保留长度为 1 的列轴，使 PyTorch 能把每一行的均值广播到该行所有列。如果结果只有 `(2,)`，与 `(2, 3)` 相减时，尾部维度 2 和 3 无法对应。

`mean` 示例使用浮点输入；当前 einops 的内置均值归约不接受普通整型张量，需要时先转换为浮点类型。

### 16.8 手算 2×2 平均池化和最大池化

`image` 是 4×4 浮点矩阵，`ph=pw=2` 为池化块大小。`average` 和 `maximum` 的返回形状都是 `(2, 2)`，分别保存每块均值和最大值。

```python
import torch
from einops import reduce

image = torch.arange(16, dtype=torch.float32).reshape(4, 4)
average = reduce(image, "(h ph) (w pw) -> h w", "mean", ph=2, pw=2)
maximum = reduce(image, "(h ph) (w pw) -> h w", "max", ph=2, pw=2)

assert average.tolist() == [[2.5, 4.5], [10.5, 12.5]]
assert maximum.tolist() == [[5., 7.], [13., 15.]]
```

例如左上块为 `[[0, 1], [4, 5]]`，均值为 `(0+1+4+5)/4=2.5`，最大值为 5。这里是不重叠、步长等于块大小、没有 padding 的池化，不能直接代表任意滑动窗口池化。

### 16.9 布尔归约：每个样本是否包含有效 token

`valid` 是 `(batch, seq)` 的布尔 mask，`True` 表示有效位置。`has_valid` 返回每个样本是否至少有一个有效位置，`all_valid` 返回是否全部有效。

```python
import torch
from einops import reduce

valid = torch.tensor([[True, False, True], [False, False, False]])
has_valid = reduce(valid, "batch seq -> batch", "any")
all_valid = reduce(valid, "batch seq -> batch", "all")

assert has_valid.tolist() == [True, False]
assert all_valid.tolist() == [False, False]
```

`seq` 在右侧消失，所以对每个样本的序列位置做逻辑归约。`any` 相当于逻辑“或”，`all` 相当于逻辑“且”。

### 16.10 自定义归约：稳定的 log-sum-exp

`x` 每行有两个分数。`logsumexp` 接收 einops 传来的张量 `tensor` 和待归约轴元组 `axes`，返回在这些轴上计算的 log-sum-exp；最终 `y` 每行保留一个结果。

```python
import torch
from einops import reduce

def logsumexp(tensor, axes):
    return torch.logsumexp(tensor, dim=axes)

x = torch.tensor([[0., 0.], [1000., 1000.]])
y = reduce(x, "batch choices -> batch", logsumexp)

expected = torch.tensor([0., 1000.]) + torch.log(torch.tensor(2.))
assert torch.allclose(y, expected)
```

先定义兼容接口，再把函数本身作为第三个参数传入。einops 可能先重新排列张量，因此自定义函数应该使用传入的 `axes`，不要固定写 `dim=1`。`torch.logsumexp` 负责稳定计算，避免直接对 1000 求指数带来的溢出。

### 16.11 用 `repeat` 放大一张小图

`image` 是 2×2 图像。`rh/rw` 分别表示每个像素在高和宽方向重复的次数。`enlarged` 返回 4×6 图像。

```python
import torch
from einops import repeat

image = torch.tensor([[1, 2], [3, 4]])
enlarged = repeat(image, "h w -> (h rh) (w rw)", rh=2, rw=3)

assert enlarged.tolist() == [
    [1, 1, 1, 2, 2, 2],
    [1, 1, 1, 2, 2, 2],
    [3, 3, 3, 4, 4, 4],
    [3, 3, 3, 4, 4, 4],
]
```

`(h rh)` 让每一行连续重复两次，`(w rw)` 让每个像素连续重复三次。改成 `(rh h)` 则会重复整幅图的行序列，含义就变了。

### 16.12 重复后归约：哪些操作能恢复原值

`x` 是两个浮点数，`copies` 为三份副本。`averaged` 对副本轴求平均，`summed` 对副本轴求和，两者都返回长度为 2 的张量。

```python
import torch
from einops import reduce, repeat

x = torch.tensor([2., 5.])
copies = repeat(x, "feature -> copy feature", copy=3)
averaged = reduce(copies, "copy feature -> feature", "mean")
summed = reduce(copies, "copy feature -> feature", "sum")

assert torch.equal(averaged, x)
assert summed.tolist() == [6., 15.]
```

三份完全相同的数据取平均可恢复原值，求和则变为原值的三倍。这只说明对重复生成的数据有这种性质；一般的 `reduce` 会丢失信息，不能靠 `repeat` 还原任意原始输入。

### 16.13 `einsum` 外积：共享名字与不同名字的区别

`x` 长度为 2，`y` 长度为 3；`row` 和 `col` 是不同轴。`outer` 返回 `(2, 3)` 矩阵，每个元素等于 `x[row] * y[col]`。

```python
import torch
from einops import einsum

x = torch.tensor([1., 2.])
y = torch.tensor([10., 20., 30.])
outer = einsum(x, y, "row, col -> row col")

assert outer.tolist() == [[10., 20., 30.], [20., 40., 60.]]
```

两个输入轴都出现在右侧，所以没有求和。与点积的 `"dim, dim ->"` 对比：点积使用相同名字对齐，并在输出端删除该名字。

### 16.14 手算矩阵乘法

`a` 和 `b` 都是 2×2 矩阵；`inner` 是求和轴，`row/col` 是保留轴。`c` 为矩阵乘法的返回结果。

```python
import torch
from einops import einsum

a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[5., 6.], [7., 8.]])
c = einsum(a, b, "row inner, inner col -> row col")

assert c.tolist() == [[19., 22.], [43., 50.]]
assert torch.equal(c, a @ b)
```

例如 `c[0, 1] = 1*6 + 2*8 = 22`。在表达式中找出“输入共享且输出消失”的 `inner`，就能找到这里的乘加方向。

### 16.15 逐元素乘法与加权平均

`values` 为两个样本、各三个数，`weights` 为三个位置的权重，且权重之和为 1。`weighted` 保留逐位置乘积；`averages` 进一步沿位置轴相加。

```python
import torch
from einops import einsum

values = torch.tensor([[10., 20., 30.], [40., 50., 60.]])
weights = torch.tensor([0.2, 0.3, 0.5])
weighted = einsum(values, weights, "batch seq, seq -> batch seq")
averages = einsum(values, weights, "batch seq, seq -> batch")

assert torch.allclose(weighted, torch.tensor([[2., 6., 15.], [8., 15., 30.]]))
assert torch.allclose(averages, torch.tensor([23., 53.]))
```

第一行加权平均为 `10*0.2 + 20*0.3 + 30*0.5 = 23`。`einsum` 不会自动让权重和为 1；如果传入未归一化的权重，得到的就是加权和。

### 16.16 `pack` 中的 `*` 可以匹配零个轴

`cls` 的形状为 `(batch, dim)`，没有序列轴；`tokens` 为 `(batch, seq, dim)`。`packed` 把每个样本的 `cls` 当作一个位置，返回 `(1, 3, 2)`；`shapes` 保存结构元数据。

```python
import torch
from einops import pack, unpack

cls = torch.tensor([[90, 91]])
tokens = torch.tensor([[[1, 2], [3, 4]]])
packed, shapes = pack([cls, tokens], "batch * dim")
cls_out, tokens_out = unpack(packed, shapes, "batch * dim")

assert [tuple(s) for s in shapes] == [(), (2,)]
assert packed.tolist() == [[[90, 91], [1, 2], [3, 4]]]
assert cls_out.shape == (1, 2)
assert torch.equal(cls_out, cls)
assert torch.equal(tokens_out, tokens)
```

对 `cls` 而言，`*` 匹配空的轴列表 `()`；零个轴的长度乘积是 1，因此仍占一个打包位置。`unpack` 会去掉这个临时位置轴，恢复原来的二维形状。

### 16.17 不同长度的序列沿同一轴打包

`short` 有两个 token，`long` 有三个 token，每个 token 两个特征。两者共有 `dim=2`，`packed` 返回五个 token，`restored` 是包含两个恢复张量的列表。

```python
import torch
from einops import pack, unpack

short = torch.tensor([[1, 2], [3, 4]])
long = torch.tensor([[5, 6], [7, 8], [9, 10]])
packed, shapes = pack([short, long], "* dim")
restored = unpack(packed, shapes, "* dim")

assert packed.shape == (5, 2)
assert torch.equal(packed, torch.cat([short, long], dim=0))
assert torch.equal(restored[0], short)
assert torch.equal(restored[1], long)
```

`pack` 记录边界，但不会自动生成 attention mask。如果这两段代表独立样本，后续直接对五个 token 做注意力会让它们相互影响，需要由模型逻辑显式限制。

### 16.18 `parse_shape` 读取尾部特征数

`x` 为 `(2, 3, 4, 5)`。`...` 匹配任意数量的前导轴，`dim` 命名最后一个轴。返回字典 `tail` 只包含最后一维；`selected` 额外保留首维。

```python
import torch
from einops import parse_shape

x = torch.zeros(2, 3, 4, 5)
tail = parse_shape(x, "... dim")
selected = parse_shape(x, "batch ... dim")

assert tail == {"dim": 5}
assert selected == {"batch": 2, "dim": 5}
```

`parse_shape` 不改变张量，也不会给 `...` 覆盖的维度逐个起名。因此这里适合提取固定位置的尺寸，不能用于保存省略部分的完整结构。

### 16.19 Layer 版本实际运行并检查结果

`model` 是一个 `nn.Sequential`，先将每张图片展平成特征向量，再对特征求平均。`x` 是两张 2×2 单通道图；返回 `y` 为两个图像均值。

```python
import torch
from torch import nn
from einops.layers.torch import Rearrange, Reduce

model = nn.Sequential(
    Rearrange("batch channel height width -> batch (channel height width)"),
    Reduce("batch feature -> batch", "mean"),
)
x = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2)
y = model(x)

assert y.tolist() == [1.5, 5.5]
assert list(model.parameters()) == []
```

第一层输出 `[[0,1,2,3], [4,5,6,7]]`，第二层分别取均值。这两个层没有可训练参数；作为 `nn.Module` 并不意味着一定包含权重。

### 16.20 梯度会怎样穿过重复操作

`x` 包含两个需要梯度的浮点数，`copies` 是三份副本，`loss` 为所有副本元素的和。`x.grad` 保存反向传播得到的梯度。

```python
import torch
from einops import reduce, repeat

x = torch.tensor([2., 5.], requires_grad=True)
copies = repeat(x, "feature -> copy feature", copy=3)
loss = reduce(copies, "copy feature ->", "sum")
loss.backward()

assert loss.item() == 21.
assert x.grad.tolist() == [3., 3.]
```

每个原始元素参与了三次求和，所以对应梯度为 3。einops 通过后端张量操作保持这条计算图；实际可微性仍取决于具体算子和输入类型。

## 17. 补充练习：先预测，再展开答案

### 17.1 区分块重复与元素重复

给定 `x = torch.tensor([4, 7, 9])`，预测下面两个返回结果。

```text
repeat(x, "n -> (r n)", r=2)
repeat(x, "n -> (n r)", r=2)
```

<details>
<summary>答案与原因</summary>

第一种为 `[4, 7, 9, 4, 7, 9]`，每份副本先走完 `n`；第二种为 `[4, 4, 7, 7, 9, 9]`，每个元素先走完重复轴 `r`。

</details>

### 17.2 多头输出拼接时哪个写法正确

`context` 的轴为 `(batch, head, seq, dim)`，目标是每个 token 合并所有头的特征。下面哪种符合目标？

```text
A: b h s d -> b s (h d)
B: b h s d -> b s (d h)
```

<details>
<summary>答案与原因</summary>

A 按头依次拼接每个头的全部特征，通常与 `b s (h d) -> b h s d` 的拆分对应。B 会交错排列各个头相同特征位置的元素。两者 shape 相同，不能只看 shape 判断正确性。

</details>

### 17.3 分组求和

给定 `x = torch.arange(8)`，希望返回 `[1, 5, 9, 13]`，如何写 pattern？

<details>
<summary>答案与原因</summary>

`reduce(x, "(group pair) -> group", "sum", pair=2)`。

先分成 `[0,1]`、`[2,3]`、`[4,5]`、`[6,7]`，再删除 `pair` 轴并求和，保留四个组。

</details>

### 17.4 同一个 batch 内每两个向量的相似度

`x.shape == (2, 5, 3)`，分别表示 batch、向量数量、特征数。要求每个 batch 内的五个向量两两做点积，输出 `(2, 5, 5)`。

<details>
<summary>答案与原因</summary>

`einsum(x, x, "batch i dim, batch j dim -> batch i j")`。

把同一张量作为两个输入，用 `i` 和 `j` 区分两次选择的向量，沿 `dim` 求和。保留 `batch`，确保不同样本不混合。

</details>

## 18. 最值得记住的要点

- pattern 左侧描述输入，右侧描述输出。
- 轴名表达语义，不是 Python 变量。
- 括号表示轴的合并或拆分，括号内部顺序会影响元素排列。
- `rearrange` 保持元素总数不变。
- `reduce` 中只在输入端出现的轴会被归约。
- `repeat` 可以创建并扩展新轴，新轴长度必须给出。
- `einsum` 中没有出现在输出端的索引轴会被求和，pattern 放在张量参数之后。
- `pack` 返回的 `packed_shapes` 是可靠执行 `unpack` 的关键元数据。
- 复杂变换先手算 shape，再用 `torch.arange` 检查元素顺序。

## 19. 参考资料

- [einops 官方主页与 API 概览](https://einops.rocks/)
- [官方教程：Einops basics](https://einops.rocks/docs/1-einops-basics/)
- [官方 API：einsum](https://einops.rocks/api/einsum/)
- [官方 API：parse_shape](https://einops.rocks/api/parse_shape/)
- [官方 GitHub 仓库](https://github.com/arogozhnikov/einops)
