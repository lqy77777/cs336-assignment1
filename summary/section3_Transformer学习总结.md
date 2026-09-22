# Section 3：Transformer 学习总结

> 整理日期：2026-09-14。  
> 对应代码文件：[cs336_basics/transformer.py](../cs336_basics/transformer.py)。本文会随 Section 3 的学习和实际实现逐条补充。  
> 当前范围：“为什么 Linear 类要继承 nn.Module”、注意力中 Q、K、V 的形状设计、QKV 的原理和含义，以及 RoPE 旋转矩阵中的索引含义。各条中的验证情况单独说明。

## 1. 为什么 Linear 类要继承 `nn.Module`

### 1.1 整体思路

Linear 层除了计算线性变换，还需要让训练流程能够找到、更新、保存和移动它的权重。继承 `nn.Module`，就是采用 PyTorch 提供的模型组件管理机制。

本作业的 Linear 不带 bias：若输入最后一维是 `d_in`，权重 `W` 的形状为 `(d_out, d_in)`，则计算 `y = x @ W.T`，输出最后一维变成 `d_out`，前导维度保持不变。

其中，矩阵乘法决定“怎么算”，`nn.Module` 负责“如何组织和管理这一层”。

### 1.2 先区分几个名字与接口

| 名称或调用 | 含义及返回值 |
| --- | --- |
| `self` | 当前创建的 Linear 实例 |
| `nn.Module` | PyTorch 神经网络模块的基类，提供参数、子模块、设备和状态管理 |
| `super().__init__()` | 调用父类初始化方法，建立模块管理参数和子模块的内部结构；不返回模型输出 |
| `nn.Parameter` | 特殊的 Tensor 类型；赋给已初始化模块的属性时，会自动登记为参数，默认需要梯度 |
| `self.weight` | 预期用于保存 Linear 权重的属性名；需要赋值为 `nn.Parameter` 才会自动登记为参数 |
| `forward(self, x)` | 定义输入 `x` 如何变成输出张量 `y` |
| `layer(x)` | 经过模块的调用机制执行 `forward`，返回层的输出 |
| `layer.parameters()` | 返回参数迭代器，默认递归包含子模块的参数，可交给优化器 |
| `layer.state_dict()` | 返回按名称组织的状态字典，包含参数和持久化 buffer；不是保存到磁盘的操作本身 |
| `layer.load_state_dict(weights)` | 将给定状态加载进模块；默认严格检查键，返回缺失键和意外键等加载检查信息 |

### 1.3 初始化与使用时发生了什么

**第一步：初始化父类。**

在自己的 `__init__` 中，先调用：

```python
super().__init__()
```

`super()` 访问父类方法，`__init__()` 建立 `nn.Module` 的内部管理结构。登记参数和子模块之前必须完成这一步，否则赋值 `nn.Parameter` 等操作会报错。

**第二步：把权重登记为参数。**

当一个 `nn.Parameter` 被赋给 `self.weight` 时，`nn.Module` 会自动记录它。后续通过 `named_parameters()` 可以看到参数名与参数张量，通过 `parameters()` 可以遍历参数张量。

普通 Tensor 即使设置了 `requires_grad=True`，直接赋给模块属性也不会自动成为已登记参数。它可以参与自动求导，但不会因此出现在 `layer.parameters()` 中。

**第三步：在 `forward` 中描述计算。**

`forward` 接收输入张量，使用权重计算并返回输出。实际使用通常写 `layer(x)`：模块的调用机制会执行 `forward`，并处理 hooks 等功能。直接调用 `layer.forward(x)` 会绕过模块调用机制中的部分功能。

**第四步：接入整个模型和训练流程。**

当 Linear 作为一个模块属性放入外层模型时，它会被登记为子模块。外层模型的 `parameters()` 默认递归查找，因此可以找到 Linear 的权重，优化器不需要逐层手工收集参数。

设备、类型和状态管理也沿子模块递归进行：

- `model.to(device="cuda")`：将已登记参数和 buffer 移到 GPU，前提是运行环境支持 CUDA。
- `model.to(dtype=torch.float32)`：将适用的浮点或复数参数及 buffer 转换到指定类型；不会把整数 buffer 随意转成浮点数。
- `model.state_dict()`：收集模型及子模块的参数和持久化 buffer，以便后续保存。
- `model.load_state_dict(weights)`：将保存或测试提供的状态加载回来。

任意普通 Tensor 属性不会自动得到与已登记参数、buffer 相同的管理。模块的 `.to(...)` 也不会自动移动以后传入的输入 `x`，输入设备仍需匹配。

### 1.4 小例子：从参数名理解管理关系

假设以后实现一个 `in_features=3`、`out_features=2` 的 Linear，并将权重登记为 `self.weight`，则预期：

```text
输入 x：               (..., 3)
权重 layer.weight：    (2, 3)
输出 layer(x)：        (..., 2)

layer.named_parameters()：包含名为 weight 的参数
layer.state_dict()：     包含键 weight
```

如果外层模型把这一层保存在 `self.proj`，外层状态字典中的相应键通常就变为 `proj.weight`。这让模型能按结构组织权重，也便于作业 adapter 将测试权重加载到正确的位置。

这是对接口行为的说明，尚不代表当前项目中已经实现或验证了这些功能。

### 1.5 容易混淆的两点

**继承 `nn.Module` 不等于自动实现线性变换。**

你仍需定义权重的形状、初始化方式以及 `forward` 中的计算。`nn.Module` 不会自动知道这层应该做矩阵乘法。

**自动求导不要求继承 `nn.Module`。**

Tensor 的 autograd 机制负责记录运算和计算梯度。只要相关 Tensor 需要梯度、运算可微且没有切断计算图，就可以反向传播。`nn.Module` 提供的是组织和管理能力；它也不会自己更新权重，参数更新通常由优化器负责。

### 1.6 写完后如何自查

后续完成 Linear 时，可以依次检查：

1. 是否在登记参数之前调用了 `super().__init__()`。
2. 权重是否为 `nn.Parameter`，形状是否为 `(out_features, in_features)`。
3. `dict(layer.named_parameters())` 是否包含预期的权重名。
4. `layer.state_dict()` 的键是否与 adapter 加载权重的方式匹配。
5. `layer(x)` 的输出是否保留前导维度，仅改变最后一维。

值得记住：`nn.Module` 管理层，`nn.Parameter` 登记权重，`forward` 定义计算，autograd 计算梯度，优化器更新参数。

## 2. 为什么 Q、K、V 的形状这样设计

### 2.1 参数、变量与返回值

Scaled dot-product attention 的输入和输出为：

$$
Q\in\mathbb{R}^{n\times d_k},\qquad
K\in\mathbb{R}^{m\times d_k},\qquad
V\in\mathbb{R}^{m\times d_v}
$$

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\in\mathbb{R}^{n\times d_v}
$$

各个符号的含义是：

| 符号 | 含义 |
| --- | --- |
| `n` | query 的数量，也就是需要生成多少个输出位置 |
| `m` | key-value 条目的数量，也就是每个 query 可以查询多少个信息位置 |
| `d_k` | 每个 query 和 key 向量的维度；二者必须相同，才能做点积 |
| `d_v` | 每个 value 向量的维度，也是 attention 输出的最后一维 |
| `Q` | `n` 个 query 向量，每行描述一个位置正在寻找什么信息 |
| `K` | `m` 个 key 向量，每行描述一个候选位置可用什么特征被匹配 |
| `V` | 与 `m` 个 key 一一对应的 value 向量，保存匹配成功后实际取出的信息 |
| 返回值 | `n` 个输出向量；每个 query 得到一个 `d_v` 维的 value 加权和 |

这里的 `Q`、`K`、`V` 是 attention 操作的输入张量，不是投影层的权重矩阵。它们通常由输入激活经过 `W_Q`、`W_K`、`W_V` 投影得到。

### 2.2 从矩阵乘法推导形状

第一步计算 query 和所有 key 的相似度：

$$
QK^\top:\quad
(n\times d_k)(d_k\times m)\longrightarrow n\times m
$$

矩阵乘法要求中间两个维度相同，所以 `Q` 和 `K` 的最后一维都必须是 `d_k`。结果中的第 `i` 行、第 `j` 列为：

$$
(QK^\top)_{i,j}=q_i\cdot k_j
$$

它表示第 `i` 个 query 与第 `j` 个 key 的匹配分数。因此，分数矩阵有 `n` 行、`m` 列：每个 query 都会与所有 key 比较一次。

第二步除以 `sqrt(d_k)`，再沿 key 维度做 softmax：

$$
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)
\in\mathbb{R}^{n\times m}
$$

`A[i, j]` 表示第 `i` 个 query 应该从第 `j` 个 value 取多少信息。对固定的 `i`，这一行的 `m` 个权重之和为 1：

$$
\sum_{j=1}^{m}A_{i,j}=1
$$

第三步使用这些权重混合 value：

$$
AV:\quad
(n\times m)(m\times d_v)\longrightarrow n\times d_v
$$

结果的第 `i` 行是：

$$
o_i=\sum_{j=1}^{m}A_{i,j}v_j
$$

也就是第 `i` 个 query 根据自己的注意力权重，对所有 `m` 个 value 做加权求和。

完整的形状变化为：

```text
Q：      (n, d_k)
K：      (m, d_k)
K.T：    (d_k, m)
               │
Q @ K.T ───────┘ → scores：(n, m)
                         │ softmax（沿 m 个 key）
                         ↓
                    weights：(n, m)
V：                         (m, d_v)
                            │
weights @ V ────────────────┘ → output：(n, d_v)
```

### 2.3 为什么 Q 和 K 的最后一维必须相同

query 和 key 要通过点积计算匹配程度：

$$
q_i\cdot k_j=\sum_{r=1}^{d_k}q_{i,r}k_{j,r}
$$

两个向量必须有相同数量的分量，才能逐分量相乘并求和。因此二者都使用 `d_k`。

这类似于搜索：query 表示“我想找什么”，key 表示“我可以用哪些特征被检索”。只有把它们表示在同一个匹配空间中，点积才有明确含义。

### 2.4 为什么 K 和 V 都有 m 行

每个 key 都必须对应一个 value：

```text
key_0 ↔ value_0
key_1 ↔ value_1
...
key_(m-1) ↔ value_(m-1)
```

query 与 `key_j` 的匹配分数最终会成为 `value_j` 的权重。如果 K 有 `m` 行而 V 有另一种条目数量，这些权重就无法对应 value，矩阵乘法 `(n, m) @ (?, d_v)` 也无法成立。

K 和 V 的行数相同，不代表其每行数值或最后一维相同。K 用于“匹配”，V 用于“传递信息”，所以它们承担不同职责。

### 2.5 为什么 V 可以使用不同的 d_v

`d_k` 只参与 query-key 的相似度计算。计算出 `n × m` 个权重后，attention 对 value 加权求和，因此 value 可以处于另一个维度为 `d_v` 的信息空间。

```text
d_k：决定用多少个特征进行匹配
d_v：决定每次取回多少个特征，以及输出向量有多宽
```

所以 `d_v` 在数学上不必等于 `d_k`。不过，本作业的多头自注意力按照 Transformer 的常见设置取：

$$
d_k=d_v=\frac{d_{model}}{h}
$$

这样 `h` 个 head 的输出拼接后，宽度正好回到 `d_model`。

### 2.6 为什么 n 和 m 可以不同

`n` 是提出问题的 query 数量，`m` 是可供查询的 key-value 条目数量。attention 的通用定义允许二者不同。

- 自注意力中，Q、K、V 通常来自同一序列，所以一般 `n = m = sequence_length`。
- 交叉注意力中，Q 可以来自目标序列，K 和 V 来自源序列，因此两个序列长度可以不同。
- 增量生成时，可能只有一个新 query 查询缓存中的许多 key 和 value，此时可以出现 `n = 1`、`m` 很大。

当前作业的通用 scaled dot-product attention 测试也区分 query 数和 key 数：测试数据中 `n_queries=12`、`n_keys=16`，因此实现不应假定二者总是相同。到了 causal self-attention，由于 Q、K、V 来自同一个输入序列，它们的序列长度才会相同。

### 2.7 具体例子：两个 query 查询三个 key-value 条目

设：

```text
n = 2：有两个 query
m = 3：有三个可查询的位置
d_k = 2：query 和 key 都用两个分量进行匹配
d_v = 4：每个 value 携带四个分量的信息
```

因此：

```text
Q.shape = (2, 2)
K.shape = (3, 2)
V.shape = (3, 4)
```

形状变化为：

```text
Q @ K.T：       (2, 2) @ (2, 3) → (2, 3)
softmax 后：    每个 query 对三个 key 得到三个权重
weights @ V：   (2, 3) @ (3, 4) → (2, 4)
```

假设 softmax 后的权重为：

$$
A=
\begin{bmatrix}
0.7&0.2&0.1\\
0.1&0.3&0.6
\end{bmatrix}
$$

那么两个 query 的输出分别为：

$$
o_1=0.7v_1+0.2v_2+0.1v_3
$$

$$
o_2=0.1v_1+0.3v_2+0.6v_3
$$

两个 query 都查看相同的三个 key-value 条目，但根据自己的 query 得到不同权重，所以最终取回的信息不同。

### 2.8 加上 batch 和多头维度后

实际代码需要支持额外的 batch-like 维度。多头注意力中常见形状为：

```text
Q：(batch, head, n, d_k)
K：(batch, head, m, d_k)
V：(batch, head, m, d_v)
```

`batch` 和 `head` 都是独立批处理的维度；每个 batch 中的每个 head 分别执行相同的注意力计算：

```text
scores： (batch, head, n, m)
output： (batch, head, n, d_v)
```

在 einops 中，核心计算可以用轴名理解为：

```python
scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key")
output = einsum(weights, V, "... query key, ... key d_v -> ... query d_v")
```

这里 `...` 代表任意数量的前导 batch-like 维度。上述代码片段用于解释形状关系，不表示当前项目中的 attention 已完成实现。

### 2.9 最容易写错的地方

- `Q @ K.T` 的输出是 `(n, m)`：行对应 query，列对应 key。
- softmax 应沿 key 轴进行，使每个 query 分配给所有 key 的权重之和为 1。
- mask 要与 query-key 分数对应，末尾两个维度应能广播到 `(n, m)`。
- K 和 V 必须拥有相同的条目数 `m`，因为一个 key 的权重用于取出同位置的 value。
- 输出数量由 Q 决定，所以输出有 `n` 行；输出宽度由 V 决定，所以最后一维为 `d_v`。

值得记住：Q 决定“谁在查询”和输出数量，K 决定“按什么特征匹配”，V 决定“匹配后取回什么”以及输出宽度。

## 3. Q、K、V 的原理和含义

### 3.1 先理解 attention 想解决什么问题

输入 Transformer 的每个 token 一开始只有自己的表示，但理解一个 token 往往需要参考序列中的其他 token。Attention 的任务是：**对于每个当前位置，判断其他位置分别有多相关，再按相关程度收集它们的信息。**

对于位置 `i`，这一过程可以拆成三个问题：

```text
Q（Query）：位置 i 现在想寻找什么信息？
K（Key）：  每个候选位置可以通过什么特征被匹配？
V（Value）：匹配到这个位置后，实际取回什么信息？
```

Q 和 K 共同决定“关注谁”，V 决定“从被关注的位置读取什么”。

### 3.2 Q、K、V 从哪里来

设输入序列的激活为：

$$
X\in\mathbb{R}^{n\times d_{model}}
$$

单头情形下，使用三组不同的可学习参数进行投影：

$$
Q=XW_Q^\top,\qquad K=XW_K^\top,\qquad V=XW_V^\top
$$

若遵循本作业中 Linear 权重 `(d_out, d_in)` 的存储方式，则：

```text
X：   (n, d_model)
W_Q： (d_k, d_model)  → Q：(n, d_k)
W_K： (d_k, d_model)  → K：(n, d_k)
W_V： (d_v, d_model)  → V：(n, d_v)
```

这里：

| 名称 | 含义 |
| --- | --- |
| `X` | 当前层收到的 token 表示 |
| `W_Q` | 将输入投影为“查询特征”的可学习权重 |
| `W_K` | 将输入投影为“匹配特征”的可学习权重 |
| `W_V` | 将输入投影为“待传递内容”的可学习权重 |
| `Q`、`K`、`V` | 由当前输入动态计算出的张量；输入改变时，它们也会改变 |

因此，QKV 不是人为给每个维度规定好“主语”“动词”等含义，也不是三份固定数据。模型通过训练逐渐学会怎样构造适合任务的 query、key 和 value 表示。

### 3.3 为什么同一个输入要投影成三份

同一个 token 在一次注意力中同时承担三种角色：

1. 它作为 query，决定自己需要从上下文找什么。
2. 它作为 key，决定其他 query 可以怎样匹配到自己。
3. 它作为 value，决定其他位置关注自己时能取得什么内容。

如果三者完全使用同一份表示，模型就必须用完全相同的特征完成“提出需求”“被检索”和“传递内容”三件不同的事。三组投影让模型可以分别学习这些功能。

`W_Q` 和 `W_K` 分开还有一个效果：位置 `i` 对位置 `j` 的注意力不必与位置 `j` 对位置 `i` 的注意力相同。一般来说：

$$
q_i\cdot k_j\neq q_j\cdot k_i
$$

语言关系本来就可能有方向。例如，代词寻找先行词与名词寻找后续修饰信息不是同一件事。

### 3.4 Q 和 K 如何决定关注谁

对于第 `i` 个 query 和第 `j` 个 key，先计算点积：

$$
s_{i,j}=q_i\cdot k_j
$$

点积越大，表示模型当前认为 `q_i` 与 `k_j` 越匹配。把所有位置组合起来，就得到 query-key 分数矩阵：

$$
S=QK^\top
$$

分数还会除以 `sqrt(d_k)`：

$$
\hat S=\frac{QK^\top}{\sqrt{d_k}}
$$

当 `d_k` 较大时，点积是许多乘积之和，数值幅度容易随维度增大。缩放可以避免 softmax 输入过大、分布过早变得极端，有利于保持训练稳定。

随后对每个 query 对应的一行沿 key 维度做 softmax：

$$
A_{i,j}=\frac{\exp(\hat S_{i,j})}{\sum_t\exp(\hat S_{i,t})}
$$

`A[i, j]` 就是第 `i` 个位置分给第 `j` 个位置的注意力权重。固定 `i` 时，所有允许访问的 key 权重之和为 1。

### 3.5 V 如何提供真正的输出内容

Q 和 K 只生成权重，本身不是最终收集的内容。得到注意力权重后，用它们对 V 加权求和：

$$
o_i=\sum_j A_{i,j}v_j
$$

例如，第一个 query 对三个位置的权重为：

```text
[0.1, 0.7, 0.2]
```

那么它的输出为：

$$
o_1=0.1v_1+0.7v_2+0.2v_3
$$

第二个位置的 value 占比最大，所以它对输出影响最大。输出通常不是复制某一个 value，而是多个 value 的加权组合。

这也说明为什么 K 与 V 要分开：模型可能使用某些特征判断一个位置是否相关，但真正传递给输出的可以是另一组特征。可以把 K 理解为索引或地址，把 V 理解为地址对应的内容；这个类比用于帮助理解，但神经网络中的 K 和 V 都是连续向量，并由训练学习得到。

### 3.6 一个句子中的直觉例子

考虑句子：

```text
小猫看到食物，于是它跑了过去。
```

当模型处理“它”时，可以用下面的方式建立直觉：

```text
“它”的 query：寻找一个可能作为指代对象的前文位置
“小猫”的 key：包含可能被代词匹配到的特征
“小猫”的 value：包含要传递给“它”的上下文信息
```

如果 `q_它 · k_小猫` 得到较高分数，softmax 就会给“小猫”的 value 较大权重，使“它”的新表示吸收更多与“小猫”有关的信息。

这只是帮助理解的例子，不能据此断言某个具体训练好的 head 一定按这种方式工作。模型学习到的注意力模式取决于数据、参数和训练过程。

### 3.7 自注意力为什么能产生上下文表示

在 self-attention 中，Q、K、V 都由同一组输入 token 表示 X 产生：

```text
                  ┌─ W_Q → Q ─┐
输入 X ───────────┼─ W_K → K ─┼→ attention → 上下文化后的输出
                  └─ W_V → V ─┘
```

虽然三者来源相同，但投影权重不同，所以角色不同。每个位置都产生自己的 query，同时也向其他位置提供 key 和 value。最终，每个输出位置都是根据自己 query 得到的权重，对允许访问的 value 进行组合。

Attention 输出还会经过输出投影，并通过残差连接加回原来的 residual stream。因此，一个 Transformer block 既保留当前位置已有的信息，又加入从上下文聚合的新信息。

### 3.8 causal mask 在 QKV 流程中的位置

decoder-only 语言模型不能让当前位置看到未来 token。Causal mask 不是直接修改 Q、K 或 V，而是在 softmax 之前修改 query-key 分数：

```text
Q、K → 相似度分数 → 屏蔽未来位置 → softmax → 权重 → 加权 V
```

对于 query 位置 `i`，只允许关注满足 `j <= i` 的 key 位置。被屏蔽的分数设为负无穷，softmax 后对应权重变为 0，因此未来位置的 value 不会进入当前输出。

### 3.9 RoPE 为什么作用于 Q 和 K，而不作用于 V

RoPE 要让 query-key 匹配分数包含相对位置信息，因此它对 Q 和 K 的成对分量进行位置相关旋转。旋转后的 Q 和 K 做点积时，分数会受到两个 token 相对位置的影响。

#### 3.9.1 RoPE 旋转矩阵中的 `i` 和 `k`

作业中的单个二维旋转块写作：

$$
R_k^i=
\begin{pmatrix}
\cos(\theta_{i,k}) & -\sin(\theta_{i,k})\\
\sin(\theta_{i,k}) & \cos(\theta_{i,k})
\end{pmatrix}
$$

这里两个索引表示不同的对象：

| 索引 | 含义 | 决定了什么 |
| --- | --- | --- |
| `i` | 当前 token 在序列中的位置 | 同一组特征在当前位置总共应该旋转多少 |
| `k` | 当前处理的是向量中的第几对特征，`k \in \{1,\ldots,d_k/2\}` | 这一对特征使用哪一个旋转频率 |

因此，$R_k^i$ 可以读作：**用于序列位置 `i`、第 `k` 个特征对的二维旋转矩阵**。其中上标 `i` 不是幂，$R_k^i$ 不是“$R_k$ 的 `i` 次方”；上标和下标在这里都只是索引。

假设一个 query 或 key 向量的维度为 $d_k=6$，它会被分成三对：

```text
k = 1：第 1、2 个分量组成一对
k = 2：第 3、4 个分量组成一对
k = 3：第 5、6 个分量组成一对
```

对于位置 `i` 上的向量 $q^{(i)}$，RoPE 分别进行：

$$
\begin{pmatrix}q_1'\\q_2'\end{pmatrix}
=R_1^i\begin{pmatrix}q_1\\q_2\end{pmatrix},\qquad
\begin{pmatrix}q_3'\\q_4'\end{pmatrix}
=R_2^i\begin{pmatrix}q_3\\q_4\end{pmatrix},\qquad
\begin{pmatrix}q_5'\\q_6'\end{pmatrix}
=R_3^i\begin{pmatrix}q_5\\q_6\end{pmatrix}
$$

同一个位置 `i` 的三对分量并不会使用同一个角度，因为 `k` 不同，对应的频率不同。按照本作业的记号，旋转角可以写成：

$$
\theta_{i,k}=\frac{i}{\Theta^{(2k-2)/d_k}}
$$

- 固定 `k`、改变 `i`：观察同一特征对在不同 token 位置的旋转；位置越往后，相位随位置推进。
- 固定 `i`、改变 `k`：观察同一个 token 的不同特征对；不同特征对按照不同频率旋转。

如果在 Python 中采用从 0 开始的特征对编号 `r = 0, 1, ..., d_k/2 - 1`，同一个公式通常改写为：

$$
\theta_{p,r}=\frac{p}{\Theta^{2r/d_k}}
$$

这里 `p` 对应 token position，`r` 对应特征对编号。它和作业中从 1 开始的 `k` 是同一件事，只是索引习惯不同：$r=k-1$。实现时最容易混淆的是把 `k` 当成 token 位置，或者把 `i` 当成普通 embedding 分量编号；在这组公式中，二者分别是“位置编号”和“特征对编号”。

V 的职责是携带被取回的内容，它不参与生成匹配分数，所以本作业只对 Q、K 应用 RoPE，不对 V 应用。

整体顺序是：

```text
X → Q 投影 → 拆分 head → RoPE ─┐
X → K 投影 → 拆分 head → RoPE ─┼→ 分数 → mask → softmax → 加权 V
X → V 投影 → 拆分 head ─────────┘
```

### 3.10 多头注意力中的 QKV

多头注意力将 `d_model` 拆成 `h` 个 head。每个 head 都在较小的表示空间中独立进行 QKV 匹配与信息聚合：

```text
单个 head：Q、K、V → attention → head output
所有 head：独立计算 → 拼接 → output projection
```

不同 head 拥有不同的投影参数切片，因此可以学习不同的匹配方式。某些 head 可能更关注邻近位置，另一些可能学习较远的依赖关系；这些是可能出现的行为，不是由代码预先硬编码的规则。

在本作业中：

$$
d_k=d_v=\frac{d_{model}}{h}
$$

于是每个 head 返回 `d_v` 个特征，拼接 `h` 个 head 后得到：

$$
h\times d_v=d_{model}
$$

再通过输出投影 `W_O` 混合不同 head 的结果。

### 3.11 一次注意力的完整执行顺序

```text
1. 输入 X 分别经过 W_Q、W_K、W_V，得到 Q、K、V
2. 多头情况下，将最后一维拆为 head 和 head_dim
3. 对 Q、K 应用 RoPE，加入位置信息
4. 计算 QK^T，得到每个 query 对每个 key 的分数
5. 除以 sqrt(d_k)，控制分数尺度
6. 应用 mask，将不允许访问的位置设为负无穷
7. 沿 key 维度做 softmax，得到和为 1 的注意力权重
8. 用权重对 V 加权求和，得到每个 query 的上下文输出
9. 拼接多个 head，并经过输出投影 W_O
```

最值得记住的是：**Q 发出检索需求，K 参与计算匹配程度，V 提供最终被聚合的信息。QK 决定权重，权重再作用于 V。**
