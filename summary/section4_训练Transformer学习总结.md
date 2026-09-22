# Section 4：训练 Transformer 学习总结

> 对应代码：[cs336_basics/training.py](../cs336_basics/training.py) 和 [tests/adapters.py](../tests/adapters.py)。整理日期：2026-09-15。当前总结范围包括 cross-entropy 中根据 `target` 提取正确类别 logit 的方法，以及 optimizer 和 gradient clipping 中的 `torch.no_grad`。后续完成 Section 4 其他部分时继续补充。本文以当前实际代码为依据；测试状态会在对应位置单独说明。

## 1. 当前要解决的问题

语言模型为每个位置输出词表中所有 token 的分数：

```text
logits：(..., vocab_size)
target：(...)
```

对于每一个 batch-like 位置，需要从 `logits` 的最后一维中取出 `target` 指定的那个分数：

```text
target_logits[...,] = logits[..., target[...,]]
```

这里不能直接写 `logits[target]`。`logits[target]` 会使用 `target` 索引 `logits` 的第 0 维，本质上是在选择或重排整行；cross-entropy 需要的是“每一行根据自己的 target 选择不同的列”。

一种错误写法是：

```python
return -logits[target] + torch.log(denominator)
```

这里的错误是：`logits[target]` 没有沿最后的词表维度取值，而且产生的形状也无法和每个样本的 loss 正确对应。当前 `cross_entropy` 已经改用 `gather(dim=-1, ...)`，并对逐位置 loss 调用 `.mean()`；本次运行 `uv run pytest tests/test_nn_utils.py::test_cross_entropy -q` 已通过。

## 2. 参数、变量与返回值

| 参数或变量 | 形状 | 含义 |
| --- | --- | --- |
| `logits` | `(..., vocab_size)` | 每个样本或 token 位置对整个词表的未归一化分数 |
| `target` | `(...)` | 每个位置的正确 token ID，也就是最后一维中的列下标 |
| `target.unsqueeze(-1)` | `(..., 1)` | 为 `gather` 补出词表维度 |
| `target_logits`（保留末维） | `(..., 1)` | 每个位置取出的正确 token 分数 |
| `target_logits`（`squeeze` 后） | `(...)` | 去掉长度为 1 的末维后的正确 token 分数 |
| 最终 cross-entropy | 标量 `()` | 对所有 batch-like 位置的 loss 求平均 |

需要一直保持的形状关系是：

```python
target.shape == logits.shape[:-1]
```

## 3. 为什么 `logits[target]` 不对

考虑一个简单输入：

```python
logits = torch.tensor([
    [10, 11, 12, 13],
    [20, 21, 22, 23],
    [30, 31, 32, 33],
])

target = torch.tensor([2, 0, 1])
```

这里：

```text
logits.shape = (3, 4)
target.shape = (3,)
```

`target` 的含义是：

```text
第 0 个样本选择类别 2
第 1 个样本选择类别 0
第 2 个样本选择类别 1
```

所以希望得到：

```python
torch.tensor([
    logits[0, 2],
    logits[1, 0],
    logits[2, 1],
])

# tensor([12, 20, 31])
```

但 `logits[target]` 等价于：

```python
logits[[2, 0, 1]]
```

它选择的是第 2、0、1 行：

```python
tensor([
    [30, 31, 32, 33],
    [10, 11, 12, 13],
    [20, 21, 22, 23],
])
```

因此：

```text
logits[target]：根据 target 选择第 0 维的整行
实际需求：每一行沿最后一维选择自己的正确类别
```

## 4. 使用 `gather` 沿最后一维取值

`gather` 的基本接口是：

```python
input.gather(dim, index)
```

- `input`：被索引的 Tensor，这里是 `logits`。
- `dim`：在哪个维度选择元素；这里必须是 `-1`，即词表维度。
- `index`：每个位置要选择的下标。
- 返回值：形状与 `index` 相同。

对应写法：

```python
target_logits = logits.gather(
    dim=-1,
    index=target.unsqueeze(-1),
).squeeze(-1)
```

### 4.1 `unsqueeze(-1)` 的作用

开始时：

```text
logits.shape = (3, 4)
target.shape = (3,)
```

`gather` 的 `index` 需要与输入有相同数量的维度，因此先执行：

```python
target.unsqueeze(-1)
```

形状和内容变为：

```text
(3,) → (3, 1)

[2, 0, 1] → [[2],
              [0],
              [1]]
```

### 4.2 `gather(dim=-1)` 的选择过程

```python
selected = logits.gather(-1, target.unsqueeze(-1))
```

执行的是：

```text
selected[0, 0] = logits[0, 2]
selected[1, 0] = logits[1, 0]
selected[2, 0] = logits[2, 1]
```

结果为：

```python
tensor([
    [12],
    [20],
    [31],
])
```

### 4.3 `squeeze(-1)` 的作用

`gather` 返回值的形状与 `index` 一致，所以此时形状为 `(3, 1)`。最后执行：

```python
selected.squeeze(-1)
```

得到：

```text
(3, 1) → (3,)
```

完整形状变化：

```text
logits                       (3, 4)
target                        (3,)
target.unsqueeze(-1)          (3, 1)
gather 后                     (3, 1)
squeeze(-1) 后                (3,)
```

## 5. 为什么这种写法适合语言模型的多维输入

Transformer LM 通常输出：

```text
logits.shape = (batch_size, seq_len, vocab_size)
target.shape = (batch_size, seq_len)
```

执行：

```python
target.unsqueeze(-1)
```

得到：

```text
(batch_size, seq_len, 1)
```

然后沿最后一维 `gather`：

```text
logits： (batch_size, seq_len, vocab_size)
index：  (batch_size, seq_len, 1)
结果：   (batch_size, seq_len, 1)
```

去掉最后一维后：

```text
target_logits.shape = (batch_size, seq_len)
```

其逐元素含义是：

```text
target_logits[b, s] = logits[b, s, target[b, s]]
```

由于 `gather` 只假设词表是最后一维，所以同一写法也适用于更多 batch-like 维度：

```text
logits：(..., vocab_size)
target：(...)
```

## 6. 放回数值稳定的 cross-entropy

先对 logits 减去每个位置的最大值：

```python
max_logit = logits.max(dim=-1, keepdim=True).values
shifted_logits = logits - max_logit
```

如果从 `shifted_logits` 中提取正确类别：

```python
target_logits = shifted_logits.gather(
    dim=-1,
    index=target.unsqueeze(-1),
).squeeze(-1)
```

那么每个位置的损失可以写成：

```python
denominator = torch.exp(shifted_logits).sum(dim=-1)
losses = torch.log(denominator) - target_logits
```

最后对所有 batch-like 位置求平均：

```python
return losses.mean()
```

这里选择的是 `shifted_logits` 中的目标分数，所以最大值已经在公式两部分中抵消，不需要额外加回：

$$
-\log\operatorname{softmax}(o)_y
=\log\sum_j e^{o_j-m}-(o_y-m)
$$

当前 `training.py` 使用了与上式等价的另一种写法：从原始 `logits` 中提取目标分数，同时显式加回 `max_logit`，最后调用 `.mean()`。本次 cross-entropy 测试已通过。

## 7. 其他选择方式及取舍

二维 logits 可以使用高级索引：

```python
row = torch.arange(logits.shape[0], device=logits.device)
target_logits = logits[row, target]
```

但它只直接适合 `(batch_size, vocab_size)`。面对 `(batch_size, seq_len, vocab_size)` 时，还需要构造更多索引或先展平。

也可以构造 one-hot target 后相乘求和，但 one-hot 的形状与 logits 相同，会额外占用与 `vocab_size` 成正比的内存。

因此本题中 `gather` 更合适：不需要创建 one-hot，也能直接支持任意前置 batch 维度。

## 8. 容易忘记的边界条件

- `target` 必须是整数索引 Tensor，通常使用 `torch.long`。
- 每个 target 必须满足 `0 <= target < vocab_size`。
- `target.shape` 应等于 `logits.shape[:-1]`。
- `target` 与 `logits` 应位于兼容的设备上。
- `gather` 对 `logits` 可微，梯度会传回被选择的分数；离散索引 `target` 不需要梯度。
- 使用移动后的 `shifted_logits` 提取 target 时，不要再次加回最大值。
- 最终题目要求返回平均 loss，因此返回值应是标量，而不是 `(...)` 形状的逐位置 loss。

## 9. 验证与复习

建议先用小矩阵手算：

```python
logits = torch.tensor([
    [10.0, 11.0, 12.0, 13.0],
    [20.0, 21.0, 22.0, 23.0],
    [30.0, 31.0, 32.0, 33.0],
])
target = torch.tensor([2, 0, 1])
```

检查提取结果应为：

```python
tensor([12.0, 20.0, 31.0])
```

完成 cross-entropy 后运行：

```bash
uv run pytest -k test_cross_entropy
```

快速复习：

1. `logits[target]` 选择的是第 0 维的行，不是逐行选择目标列。
2. `target.unsqueeze(-1)` 把 `(...)` 变成 `(..., 1)`，使它能作为最后一维的索引。
3. `gather(dim=-1, ...)` 表示沿词表维度为每个位置选择自己的目标分数。
4. `gather` 的结果形状与 `index` 相同，所以最后需要 `squeeze(-1)`。
5. 对 shifted logits 使用 `gather`，可以让最大值在 cross-entropy 公式中自然抵消。

## 10. `torch.no_grad`：为什么需要、何时使用

### 10.1 先理解 autograd 在记录什么

当一个 Tensor 设置了：

```python
x.requires_grad == True
```

PyTorch 默认会记录由它参与的可微运算，建立计算图。例如：

```python
x = torch.tensor([2.0], requires_grad=True)
y = x.square()
loss = y.sum()
```

这里可以理解为：

```text
x --square--> y --sum--> loss
```

`loss.backward()` 会沿这张图反向计算，并把结果累积到：

```python
x.grad
```

训练时必须为模型的 forward 和 loss 保留这张图，否则无法计算模型参数的梯度。

### 10.2 `torch.no_grad` 的作用

`torch.no_grad` 会在一个局部范围内关闭梯度记录。常见形式有两种。

作为上下文管理器：

```python
with torch.no_grad():
    y = x * 2
```

作为函数装饰器：

```python
@torch.no_grad()
def update_parameters(...):
    ...
```

在这个范围内，即使输入 `x.requires_grad=True`，普通计算结果通常也不会连接到原计算图：

```text
y.requires_grad == False
y.grad_fn is None
```

它只临时改变“是否记录运算”的模式，不会永久修改原 Tensor：

```text
x.requires_grad 仍然是 True
x.grad 不会因此被清空
模型参数仍然是可训练参数
```

### 10.3 AdamW 的 `step()` 为什么要加

AdamW 的输入和状态包括：

| 对象 | 含义 | 是否需要对它的更新再求梯度 |
| --- | --- | --- |
| `p` | 当前模型参数 | 不需要 |
| `p.grad` / `grad` | `loss.backward()` 已算出的梯度 | 不需要 |
| `m` | 梯度的一阶矩移动平均 | 不需要 |
| `v` | 梯度平方的二阶矩移动平均 | 不需要 |
| `t` | 当前参数的更新次数 | 不需要 |

进入 `optimizer.step()` 前，训练所需的梯度已经存在 `p.grad` 中：

```text
forward → loss → backward → p.grad → optimizer.step
```

`step()` 只需要执行数值更新：

```text
m ← beta_1 m + (1-beta_1)g
v ← beta_2 v + (1-beta_2)g²
p ← p - 更新量
```

我们不会再对“optimizer 如何更新参数”求导，因此这些操作不应该成为下一张 autograd 计算图的一部分。当前实现把装饰器放在整个 `step()` 上：

```python
@torch.no_grad()
def step(self, closure=None):
    ...
```

这使 `m.mul_()`、`v.addcmul_()` 和 `p -= ...` 等原地更新不会被 autograd 记录。

### 10.4 AdamW 不加会发生什么

`p` 是 `nn.Parameter`，通常同时满足：

```text
p.requires_grad == True
p 是计算图中的叶子 Tensor
```

如果在梯度记录开启时直接执行：

```python
p -= update
```

PyTorch 一方面需要保护反向传播依赖的叶子 Tensor，另一方面又看到代码要原地覆盖它，通常会报错：

```text
RuntimeError: a leaf Variable that requires grad is being used in an in-place operation
```

即使某种写法没有立刻报错，记录 optimizer 内部的 `m`、`v` 和参数更新过程也会带来没有用途的计算图与内存开销。普通的一阶训练不需要对 optimizer update 再求导。

因此，在 optimizer 中加入 `no_grad` 同时解决两个问题：

1. 允许安全地原地修改模型参数。
2. 避免保存无用的 optimizer 更新计算图，减少内存开销。

### 10.5 加了以后会发生什么

加入 `@torch.no_grad()` 后：

- 仍然可以读取之前由 `loss.backward()` 算好的 `p.grad`。
- 不会自动清空 `p.grad`；清空梯度仍由 `optimizer.zero_grad()` 完成。
- 可以原地更新 `p`、`m` 和 `v`。
- 更新过程不会获得 `grad_fn`，也不会被下一次 backward 追踪。
- 离开 `step()` 后，外部会恢复进入函数前的梯度记录状态。

它不会让模型永久停止训练。下一轮 forward 在正常梯度模式下执行时，PyTorch 会基于更新后的参数建立一张新的计算图。

### 10.6 closure 为什么要临时重新开启梯度

当前 AdamW 的 `step()` 整体位于 `no_grad` 中，但 `closure` 的职责可能包括：

```text
重新执行 forward
重新计算 loss
调用 backward
```

这些操作必须建立计算图，所以 closure 需要局部覆盖外层的 `no_grad`：

```python
@torch.no_grad()
def step(self, closure=None):
    loss = None

    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    # 这里重新回到 no_grad，执行参数更新
    ...

    return loss
```

嵌套关系是：

```text
step：no_grad
├── closure：临时 enable_grad
└── 参数更新：no_grad
```

这也是当前 AdamW 中同时出现 `@torch.no_grad()` 和 `with torch.enable_grad()` 的原因；二者职责不同，并不矛盾。

### 10.7 gradient clipping 为什么也可以加

gradient clipping 发生在 `backward()` 之后、`step()` 之前：

```python
loss.backward()
gradient_clipping(model.parameters(), max_l2_norm)
optimizer.step()
```

它读取已经算好的 `p.grad`，然后原地缩放：

```python
grad.mul_(scale)
```

普通训练不需要对这次裁剪操作继续求导，因此也可以使用：

```python
@torch.no_grad()
def gradient_clipping(...):
    ...
```

严格来说，普通 backward 产生的 `.grad` 通常本身不要求梯度，所以某些裁剪代码即使不加也能运行；但显式使用 `no_grad` 能清楚表达“这里只修改已有梯度，不构建新的计算图”，也能避免未来代码变化后意外记录运算。

当前代码使用了 `@torch.no_grad`。在本项目当前 PyTorch 版本中该形式可以工作；更常见、更清晰的标准写法是 `@torch.no_grad()`。

### 10.8 哪些地方应该加

典型适用场景：

| 场景 | 是否使用 `no_grad` | 原因 |
| --- | --- | --- |
| optimizer 的参数更新 | 是 | 只消费现有梯度并原地修改参数 |
| 普通 gradient clipping | 是或建议使用 | 只原地修改已有 `.grad` |
| 验证与普通推理 | 是 | 不需要 backward，可减少计算图内存 |
| 手动更新 moving average | 通常是 | 统计量更新通常不参与反向传播 |
| 初始化或复制模型参数 | 通常是 | 参数赋值不应进入训练计算图 |

验证时常见写法：

```python
model.eval()
with torch.no_grad():
    logits = model(inputs)
```

其中 `model.eval()` 和 `no_grad` 解决的是两件不同的事：

- `model.eval()` 改变 Dropout、BatchNorm 等模块的行为。
- `torch.no_grad()` 关闭 autograd 记录。

两者不能互相替代。

### 10.9 哪些地方不应该加

只要后续需要通过某段计算执行 `backward()`，这段计算就不能被 `no_grad` 包住。训练时下面这些步骤不能放进 `no_grad`：

```python
logits = model(inputs)
loss = cross_entropy(logits, targets)
loss.backward()
```

错误示例：

```python
with torch.no_grad():
    logits = model(inputs)
    loss = cross_entropy(logits, targets)

loss.backward()
```

由于 `logits` 和 `loss` 没有连接到模型参数的计算图，`backward()` 无法正常向参数传播梯度，通常会报出 loss 不要求梯度或没有 `grad_fn`。

因此不能为了节省显存而把整个训练 iteration 都放入 `no_grad`。正确边界是：

```text
需要产生梯度：forward、loss、backward前的可微计算
不需要产生梯度：optimizer update、普通梯度裁剪、验证推理
```

### 10.10 `no_grad`、`detach`、`requires_grad_` 和 `eval` 的区别

| 方法 | 作用范围 | 主要作用 |
| --- | --- | --- |
| `with torch.no_grad()` | 一个代码块或被装饰函数 | 临时不记录该范围内的运算 |
| `tensor.detach()` | 一个新返回的 Tensor | 得到与原计算图断开的视图 |
| `tensor.requires_grad_(False)` | Tensor 本身的属性 | 持续关闭该 Tensor 的梯度需求，常用于冻结参数 |
| `model.eval()` | 模块的训练模式 | 改变 Dropout、BatchNorm 等行为，不负责关闭 autograd |

`no_grad` 是临时执行上下文；它不会像 `requires_grad_(False)` 那样永久冻结参数，也不会像 `detach()` 那样专门返回一个断开图的新 Tensor。

### 10.11 一个最小例子

需要梯度时：

```python
x = torch.tensor([2.0], requires_grad=True)
y = x.square()

print(y.requires_grad)  # True
print(y.grad_fn)        # 有反向传播节点

y.backward()
print(x.grad)           # tensor([4.])
```

禁用梯度记录时：

```python
x = torch.tensor([2.0], requires_grad=True)

with torch.no_grad():
    y = x.square()

print(x.requires_grad)  # True，x 本身没有被永久修改
print(y.requires_grad)  # False
print(y.grad_fn)        # None
```

这个例子的核心区别不是有没有执行乘法，而是 PyTorch 有没有为这次乘法建立反向传播路径。

### 10.12 当前实现中的位置

当前 AdamW：

```python
@torch.no_grad()
def step(self, closure=None):
```

这是必要且正确的使用位置。closure 内部又使用 `torch.enable_grad()`，边界也正确。

当前 gradient clipping：

```python
@torch.no_grad
def gradient_clipping(...):
```

它表达的意图正确；为了和 PyTorch 文档中的常见写法保持一致，可以写成 `@torch.no_grad()`。这属于表达和兼容习惯上的改进，不改变本项目当前环境下的运行意图。

### 10.13 快速判断规则

遇到一段代码时，可以问自己：

```text
我以后是否需要对这段运算的结果调用 backward，
并让梯度穿过这段运算回到输入或模型参数？
```

- 如果答案是“需要”，不要加 `no_grad`。
- 如果答案是“不需要，只是在更新、复制、统计或推理”，通常应该加。

最值得记住的四句话：

1. `no_grad` 关闭的是计算图记录，不是把已有梯度清零。
2. forward 和 loss 在训练时需要计算图，不能放在 `no_grad` 中。
3. optimizer update 不需要再求导，应该放在 `no_grad` 中。
4. 外层是 `no_grad` 时，closure 若要重新计算梯度，必须局部使用 `enable_grad`。
