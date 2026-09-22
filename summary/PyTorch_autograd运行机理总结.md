# PyTorch `autograd` 运行机理总结

> 整理日期：2026-09-16。  
> 适用环境：本项目声明 `torch~=2.11.0`，本次示例使用本机 PyTorch `2.11.0`（CPU）验证。  
> 总结范围：以 eager mode 下常用的反向模式自动微分为主，覆盖动态计算图、VJP、叶子张量、梯度累积、saved tensors、图释放、grad modes、原地操作、高阶导数、自定义 `Function` 和调试方法。  
> 资料依据：[Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)、[`torch.autograd`](https://docs.pytorch.org/docs/stable/autograd.html)、[`Tensor.backward`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html) 和 [Extending PyTorch](https://docs.pytorch.org/docs/stable/notes/extending.html)。

## 1. 先记住一张总图

PyTorch 的 `autograd` 主要使用**反向模式自动微分**。一次普通训练迭代可以概括为：

```text
带 requires_grad 的叶子张量（通常是参数）
                  │
                  ▼
前向执行算子，同时建立“怎样把上游梯度传回去”的动态图
                  │
                  ▼
得到标量 loss；loss.grad_fn 是反向图的入口
                  │
       loss.backward() 放入种子梯度 1
                  │
                  ▼
反向引擎按依赖顺序执行各 Node 的局部 VJP
                  │
        同一张量的多路梯度贡献相加
                  │
                  ▼
梯度累积到需要梯度的叶子张量 .grad
                  │
                  ▼
optimizer.step() 读取 .grad 更新参数（这不是 autograd 做的）
                  │
                  ▼
清空/置空 .grad，下一次前向重新建立一张新图
```

最重要的职责边界是：

| 组件 | 负责什么 | 不负责什么 |
| --- | --- | --- |
| `Tensor` | 保存数值、形状、dtype、device，以及 autograd 元数据 | 不自行更新参数 |
| autograd 图 | 记录当前前向计算对应的反向依赖和局部求导规则 | 不是模型结构图，也不会跨迭代永久存在 |
| autograd engine | 调度反向节点，传递并合并梯度 | 不决定优化算法 |
| `nn.Module` | 组织参数、buffer 和子模块，定义前向计算 | 本身不是自动求导引擎 |
| optimizer | 根据参数的 `.grad` 更新参数 | 通常不负责推导梯度 |

一句话概括：**前向时记录反向所需的信息，反向时从输出出发反复计算局部 VJP，并把多条路径的贡献相加到叶子张量。**

## 2. 为什么神经网络适合反向模式自动微分

设输入或参数为：

$$
x\in\mathbb{R}^{n}
$$

模型最终产生一个标量损失：

$$
L=f(x)\in\mathbb{R}
$$

深度学习常见情况是 `n` 非常大，但输出损失只有一个或少数几个。反向模式先做一次前向，再从输出向输入传播，只需共享中间结果，就可以高效得到所有参数对损失的梯度。

### 2.1 autograd 实际传播的是 VJP

对一般向量函数：

$$
y=f(x),\qquad
x\in\mathbb{R}^{n},\quad y\in\mathbb{R}^{m}
$$

Jacobian 为：

$$
J_f=\frac{\partial y}{\partial x}\in\mathbb{R}^{m\times n}
$$

PyTorch 反向传播通常不会显式构造完整的 `m × n` Jacobian，而是接收一个上游向量：

$$
v=\frac{\partial L}{\partial y}\in\mathbb{R}^{m}
$$

然后计算 vector-Jacobian product：

$$
v^T J_f=\frac{\partial L}{\partial x}
$$

这样既节省内存，也避免构造大量并不需要的 Jacobian 元素。

### 2.2 为什么标量可以直接调用 `backward()`

若 `loss` 只有一个元素，`loss.backward()` 会隐式使用种子梯度：

$$
\frac{\partial L}{\partial L}=1
$$

概念上相当于：

```python
loss.backward(torch.ones_like(loss))
```

若输出 `y` 含多个元素，就必须提供与它形状兼容的 `gradient`，说明要计算哪个线性组合的梯度：

```python
y.backward(v)
```

得到的是 `v.T @ J`，不是自动返回完整 Jacobian。完整 Jacobian、Hessian 等通常使用 `torch.func.jacrev`、`jacfwd`、`hessian`，或 `torch.autograd.functional` 中的相应工具。

## 3. 核心参数、属性和返回值

| 名称 | 含义 |
| --- | --- |
| `requires_grad` | 是否需要追踪与该张量相关、用于反向求导的计算 |
| `is_leaf` | 张量是否是 autograd 图的叶子；用户直接创建且未由被追踪运算产生的张量通常是叶子 |
| `grad_fn` | 产生该非叶子张量的反向节点入口；叶子张量通常为 `None` |
| `.grad` | 累积得到的梯度缓冲区；默认主要为需要梯度的叶子张量填充 |
| `.retain_grad()` | 要求非叶子张量在反向结束后也保留 `.grad` |
| `.backward()` | 从当前输出启动反向传播，并通常把结果累积到叶子的 `.grad` |
| `torch.autograd.grad()` | 计算并**返回**指定 outputs 对指定 inputs 的梯度，常用于精确查询、高阶导数 |
| `gradient` / `grad_outputs` | 传入反向过程的上游梯度，也就是 VJP 中的向量 `v` |
| `retain_graph` | 反向后是否保留本次图中本可释放的状态；通常不应打开 |
| `create_graph` | 是否为“梯度的计算过程”继续建图，以便再求高阶导数 |
| `detach()` | 返回与原张量共享存储、但切断当前 autograd 历史的张量 |
| `torch.no_grad()` | 暂时不记录反向图；产生的普通张量以后仍可进入 grad mode 的计算 |
| `torch.inference_mode()` | 更彻底地关闭 autograd 相关开销，但其中创建的 inference tensor 受更多限制 |
| `model.eval()` | 改变 Dropout、BatchNorm 等模块的训练/评估行为；**不会关闭 autograd** |

只有浮点和复数类型张量支持 `requires_grad=True`；整数本身不具有通常意义上的连续导数。

## 4. 前向阶段：计算数值，也动态建立反向图

考虑：

```python
x = torch.tensor([2.0, -1.0], requires_grad=True)
w = torch.tensor([3.0, 4.0], requires_grad=True)
a = x * w
b = a.square()
loss = b.sum()
```

数值计算为：

```text
x = [ 2, -1]
w = [ 3,  4]
a = x * w       = [ 6, -4]
b = a²          = [36, 16]
loss = sum(b)   = 52
```

与此同时，autograd 建立近似如下的反向关系：

```text
x（叶子） ─┐
           ├─ MulBackward ─ a ─ PowBackward ─ b ─ SumBackward ─ loss
w（叶子） ─┘
```

这里的图不是由 `nn.Module` 层组成，而是更接近**实际执行过的 Tensor 运算对应的反向节点**。一个模块的 `forward` 里可能产生许多 autograd 节点。

### 4.1 什么条件下会记录一次运算

在默认 grad mode 中，只要一次运算至少有一个输入需要梯度，并且该运算支持 autograd，它的结果一般就会带上反向历史：

```python
x.requires_grad      # True
a.requires_grad      # True
a.grad_fn            # 类似 MulBackward0
```

若所有输入都不需要梯度，或运算发生在 `no_grad` / `inference_mode` 中，就不会为该运算建立普通反向图。

### 4.2 `grad_fn` 和 Node

从 Python 层看，非叶子张量的 `grad_fn` 是进入反向图的入口。内部图的节点代表反向公式，边表示梯度应该传给哪个上游节点以及对应哪个输入槽位。

例如：

```python
type(loss.grad_fn).__name__
# 'SumBackward0'

loss.grad_fn.next_functions
# 可以继续观察反向依赖；它是调试入口，不宜作为业务逻辑依赖
```

`MulBackward0`、`PowBackward0` 这类名称属于生成的内部实现名称，可能随版本改变，不应写进需要长期稳定的程序逻辑。

### 4.3 图是动态的

PyTorch eager autograd 记录的是**这一次真正执行的路径**：

```python
if x.sum() > 0:
    y = x.square()
else:
    y = x.exp()
```

本次走哪条分支，就建立哪条分支对应的图。循环执行多少次，就记录多少次相关运算。下一次前向会根据下一次实际执行重新建图，因此 Python 控制流可以自然参与模型计算。

“动态图”不等于反向时重新执行整段 Python 前向代码。普通 backward 使用前向已建立的反向节点和保存的数据；activation checkpointing 等机制才会显式重算部分前向。

## 5. 叶子张量、非叶子张量和 `.grad`

上述例子中：

| 张量 | `is_leaf` | `requires_grad` | `grad_fn` | 默认反向后是否保存 `.grad` |
| --- | ---: | ---: | --- | --- |
| `x` | `True` | `True` | `None` | 是 |
| `w` | `True` | `True` | `None` | 是 |
| `a` | `False` | `True` | `MulBackward0` | 否 |
| `b` | `False` | `True` | `PowBackward0` | 否 |
| `loss` | `False` | `True` | `SumBackward0` | 否 |

### 5.1 为什么参数通常是叶子

`nn.Parameter` 通常由用户或模块直接创建，不是某个被追踪运算的结果，所以它是叶子。训练最关心的是损失对参数的梯度，autograd 因而默认把这些梯度写入参数 `.grad`，供优化器使用。

### 5.2 中间梯度明明算过，为什么 `.grad is None`

反向传播必须临时得到中间节点的梯度，才能继续应用链式法则，但反向使用完之后通常没有必要长期保存它。为节省内存，非叶子张量默认不填充 `.grad`。

若确实要检查中间梯度，应在 backward 前调用：

```python
a.retain_grad()
loss.backward()
print(a.grad)
```

`retain_grad()` 主要是调试或某些特殊算法的工具；大量保留激活梯度会增加内存占用。官方的[叶子与非叶子教程](https://docs.pytorch.org/tutorials/beginner/understanding_leaf_vs_nonleaf_tutorial.html)也强调，普通中间节点默认不会保留 `.grad`。

### 5.3 `requires_grad` 不是“这里已经有梯度”

它表达的是“需要追踪，以便以后可能求梯度”。刚完成前向而尚未 backward 时，一个参数可以：

```text
requires_grad = True
grad = None
```

这完全正常。

## 6. backward 到底怎样运行

以下是用户可见语义与引擎概念的结合。具体线程池、队列和节点类型属于内部实现，版本间可能变化，但链式法则和依赖调度的整体逻辑稳定。

### 6.1 第一步：从根输出放入种子梯度

标量 `loss.backward()` 从 `dL/dL = 1` 开始。若输出非标量，调用者提供 `gradient` 作为上游梯度。

### 6.2 第二步：确定需要执行的反向节点

引擎沿反向边寻找与目标输入相关的节点，并维护依赖关系。一个节点只有在所需的下游梯度贡献已经就绪后，才能安全执行。

### 6.3 第三步：每个节点执行局部 VJP

节点拿到 `grad_output`，根据本算子的局部导数算出每个需要梯度的输入对应的 `grad_input`。

对：

$$
a=x\odot w
$$

若上游梯度为 $g_a=\partial L/\partial a$，乘法节点计算：

$$
\frac{\partial L}{\partial x}=g_a\odot w
$$

$$
\frac{\partial L}{\partial w}=g_a\odot x
$$

每个节点只需知道本地运算的求导规则和收到的上游梯度，不需要单独推导整个模型的复合函数。

### 6.4 第四步：分支处把贡献相加

若一个张量被多次使用：

```python
y = x.square() + 3 * x
```

`x` 到 `y` 有两条路径。链式法则要求：

$$
\frac{dy}{dx}
=\left.\frac{dy}{dx}\right|_{x^2路径}
+\left.\frac{dy}{dx}\right|_{3x路径}
=2x+3
$$

autograd 会合并来自不同下游路径的贡献。这也是 `.grad` 使用“累积”语义的数学原因之一。

广播运算的 backward 还会把沿广播维度产生的梯度求和，使 `grad_input` 恢复为原输入形状。例如 `(B, D) + (D,)` 中，偏置的梯度会沿 batch 维求和。

### 6.5 第五步：写入叶子的 `.grad`

传播抵达需要梯度的叶子后，累积节点把结果加入其 `.grad`。对本节例子：

$$
L=\sum_i(x_iw_i)^2
$$

所以：

$$
\frac{\partial L}{\partial x_i}=2(x_iw_i)w_i
$$

$$
\frac{\partial L}{\partial w_i}=2(x_iw_i)x_i
$$

代入 `x=[2,-1]`、`w=[3,4]`：

```text
a.grad = dL/da = 2a          = [ 12, -8]  # 仅因调用了 retain_grad
x.grad = dL/dx = (2a) * w    = [ 36,-32]
w.grad = dL/dw = (2a) * x    = [ 24,  8]
```

### 6.6 第六步：释放反向所需的临时状态

默认 backward 完成后，图中为反向保存的张量等资源会在不再需要时释放。因此对同一个前向结果再次 backward，常见错误是：

```text
Trying to backward through the graph a second time ...
```

解决方向通常不是盲目添加 `retain_graph=True`，而是重新执行前向得到新图。只有确实需要在**同一张图**上做多次 VJP 时，才保留图。

## 7. 前向究竟要为反向保存什么

局部导数往往依赖前向输入或输出。例如：

$$
y=x^2,\qquad \frac{dy}{dx}=2x
$$

平方运算的 backward 需要知道前向的 `x`。PyTorch 算子会根据其导数公式保存必要张量；自定义 `torch.autograd.Function` 使用 `ctx.save_for_backward(...)` 保存。

### 7.1 保存的是“反向所需数据”，不是所有前向张量

不同运算需要的信息不同：

- 加法的输入梯度与输入数值无关，通常不必为了导数保存完整输入值。
- 乘法对一个输入求导需要另一个输入。
- `exp` 的导数可以利用输出 `exp(x)`。
- 某些 reduction 只需保存形状、维度等元数据。

因此，训练内存不等于“所有激活简单相加”，而取决于各反向公式保存了什么、张量何时最后一次被使用，以及框架是否应用重计算或保存张量 hooks。

### 7.2 `_saved_*` 只适合观察

某些 `grad_fn` 会暴露 `_saved_self`、`_saved_result` 等调试属性，可以帮助理解保存了什么。但这些以下划线开头，属于实现细节，不是稳定公共 API。

### 7.3 activation checkpointing 的本质

checkpointing 选择在前向少保存一部分激活，反向需要时重新执行相应前向区域：

```text
普通方式：更多保存激活，反向少重算
checkpoint：少保存激活，反向多重算
```

它用额外计算换显存/内存，适合“激活大、重算相对便宜”的区域，并不会改变正确配置下应得到的数学梯度。

## 8. `.backward()`、`autograd.backward()` 与 `autograd.grad()`

### 8.1 `Tensor.backward()`

```python
loss.backward()
```

最适合典型训练流程：从一个 loss 反传，把梯度累积进参数 `.grad`。

主要参数：

| 参数 | 意义 |
| --- | --- |
| `gradient` | 非标量输出所需的上游梯度 |
| `retain_graph` | 是否保留本次反向图；默认通常随 `create_graph` 推断 |
| `create_graph` | 是否记录梯度计算本身，以支持高阶导数 |
| `inputs` | 限定把梯度累积到哪些输入的 `.grad` |

### 8.2 `torch.autograd.backward()`

它是能同时从多个输出启动反向的函数式接口。`Tensor.backward()` 最终使用同一套 autograd 机制。

### 8.3 `torch.autograd.grad()`

```python
gx, gw = torch.autograd.grad(loss, (x, w))
```

它直接返回所请求的梯度，而不是以填充全部叶子 `.grad` 为主要目标，适合：

- 查询少数指定输入的梯度；
- 编写高阶导数；
- 元学习、梯度惩罚等需要把梯度继续参与计算的场景；
- 避免无关参数 `.grad` 被副作用式更新。

若 output 与某个 input 没有依赖关系，默认会报“未被图使用”的错误；可根据算法需要考虑 `allow_unused=True`。此时返回的 `None` 表示没有依赖关系，不应不加判断地等同于数值零。

## 9. 梯度为什么会累积，以及怎样清空

调用 backward 时，新梯度会加到已有 `.grad`，而不是自动覆盖：

```python
loss.backward()
loss2.backward()
# 参数 grad 是两次贡献之和（前提是两次各自的图可用）
```

这使梯度累积、多 loss 合并等做法成为可能，但也意味着普通训练迭代要显式清理：

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

`set_to_none=True` 与把已有 buffer 清零的可见状态不同：前者让 `.grad` 回到 `None`，下一次反向时再创建梯度；某些优化器还会区分“梯度为 `None`”与“梯度是全零张量”，因此选择时要理解所用优化器的行为。

如果使用梯度累积训练，可以连续做若干 micro-batch 的 backward，再调用一次 `step()`；通常还需按累积步数缩放 loss，避免梯度整体放大。

## 10. grad mode、`requires_grad`、`detach` 和 `eval` 的区别

这是最容易混淆的一组概念。

| 机制 | 是否记录反向图 | 典型用途 | 离开作用域后结果能否再进入普通求导计算 |
| --- | --- | --- | --- |
| 默认 grad mode | 根据输入 `requires_grad` 决定 | 训练前向 | 可以 |
| `requires_grad_(False)` | 从源头冻结指定叶子/参数 | 微调时冻结模块 | 张量仍是普通张量 |
| `torch.no_grad()` | 作用域内运算不记录 | optimizer 更新、临时推理 | 结果通常可以 |
| `torch.inference_mode()` | 不记录，并跳过更多 autograd 开销 | 完全不与 autograd 交互的推理/数据处理 | inference tensor 之后进入被追踪计算受到限制 |
| `detach()` | 只对返回张量切断历史 | stop-gradient、日志或分支隔离 | detach 结果可作为新起点，但原历史已断开 |
| `model.eval()` | **不影响是否记录图** | 切换 Dropout/BatchNorm 等行为 | 不适用 |

### 10.1 冻结参数

```python
for p in model.encoder.parameters():
    p.requires_grad_(False)
```

若某运算的所有相关输入都不需要梯度，autograd 可不记录这部分反向图。但如果冻结层的输入仍需要梯度，图仍可能需要穿过该层来计算对输入或更早参数的梯度；“参数不求梯度”和“整段图完全不存在”不是同一句话。

### 10.2 `detach()` 共享存储

`y = x.detach()` 通常与 `x` 共享底层存储，只是 autograd 历史被切断。修改其中一个可能影响另一个的数值并触发别名/版本相关问题。若还需要独立数据，应使用：

```python
y = x.detach().clone()
```

### 10.3 正确的评估组合

常见评估写法是：

```python
model.eval()
with torch.inference_mode():
    prediction = model(batch)
```

两者解决不同问题：`eval()` 管模块行为，`inference_mode()` 管 autograd 跟踪与开销。若输出还要在后续可微计算中使用，应考虑 `no_grad()` 或重新设计边界。

## 11. 原地操作、view、共享存储和版本计数

原地操作如 `add_()`、`relu_()` 会直接改写已有存储。它们对 autograd 困难的原因不是“名字带下划线”本身，而是：反向公式可能还需要被覆盖前的数值。

### 11.1 版本计数如何保护正确性

Tensor 会维护版本信息。某个反向节点保存张量时，也会记住对应版本。之后若原地操作修改了它，版本增加；backward 取出保存张量时发现版本不一致，就会报类似错误：

```text
one of the variables needed for gradient computation
has been modified by an inplace operation
```

这是一种正确性检查：与其静默算错梯度，PyTorch 更愿意停止。

### 11.2 叶子参数上的原地修改

在 grad mode 中直接原地修改需要梯度的叶子通常会被阻止。优化器更新参数时会使用不建立反向图的上下文，其含义是“这是训练算法的状态更新，不是当前损失函数的一部分”。

不要用 `.data` 绕开检查。它可能让 autograd 看不到修改，从而失去版本与历史保护，产生静默错误。需要修改参数或张量时，优先使用清晰的 `no_grad()`、`detach()` 和正规 optimizer API。

### 11.3 view 会共享存储

切片、转置等操作常产生 view。base tensor 与 view 共享存储，一方原地修改可能影响另一方。autograd 必须同时追踪 view 关系、别名和版本，因此复杂计算中通常优先使用 out-of-place 运算；原地操作未必带来显著内存收益，因为 autograd 本来就会积极复用和释放 buffer。

## 12. 图的生命周期：`retain_graph` 与 `create_graph`

二者名称相近，但问题不同。

### 12.1 `retain_graph=True`

含义：第一次反向结束后，不释放仍需用于再次遍历同一图的状态。

适用例子：对同一前向结果使用不同 `grad_outputs` 做多次 VJP。代价是保存的激活等资源活得更久，内存占用上升。

普通训练通常重新前向即可，不需要保留旧图。

### 12.2 `create_graph=True`

含义：把“一阶梯度怎样算出来的”也作为可微计算记录起来，从而可以继续求二阶或更高阶导数。

```python
x = torch.tensor(2.0, requires_grad=True)
y = x**3

first, = torch.autograd.grad(y, x, create_graph=True)
second, = torch.autograd.grad(first, x)

print(first)   # 12 = 3x²
print(second)  # 12 = 6x
```

高阶图会显著增加内存和计算成本。若只要普通一阶训练梯度，不应打开。

### 12.3 一个常见内存泄漏模式

若把仍连接计算图的 `loss` 或中间 Tensor 长期放入 Python list，整条关联图可能持续存活：

```python
history.append(loss)          # 可能保留图
history.append(loss.item())   # 只保存 Python 数值
history.append(loss.detach()) # 保存 Tensor 值但切断历史
```

日志记录标量时常用 `.item()`；需要批量 Tensor 数据时，明确决定是否 `detach()`，以及是否还要移到 CPU。

## 13. 自定义 `torch.autograd.Function`

只有在需要封装外部实现、写自定义算子或明确提供特殊梯度规则时，才应自定义 `Function`。若前向完全由标准 PyTorch 运算组成，直接写普通 Python 函数通常就能自动得到正确图。

示意结构：

```python
class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.square()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_x = grad_output * 2 * x
        return grad_x

square = Square.apply
```

参数与返回值：

| 项目 | 含义 |
| --- | --- |
| `ctx` | 本次调用专用的上下文，用于前向向反向传递必要信息 |
| `save_for_backward` | 保存反向需要的 Tensor，让 autograd 正确管理生命周期与检查原地修改 |
| `grad_output` | 损失对当前 Function 输出的上游梯度 |
| `backward` 返回值 | 与 `forward` 输入逐项对应的梯度；不需要梯度的项返回 `None` |

`backward` 必须把上游梯度乘进去，因为它实现的是 VJP，不只是裸局部导数。多输入、多输出时，参数和返回梯度的数量、顺序必须与接口对应。

若希望兼容 `torch.func` 的变换组合，需要遵循当前官方对 `forward` / `setup_context`、`backward`、`jvp` 和 `vmap` 的额外要求；这部分接口仍可能演进，应以对应版本文档为准。

### 13.1 用 `gradcheck` 验证

```python
x = torch.randn(3, dtype=torch.double, requires_grad=True)
torch.autograd.gradcheck(Square.apply, (x,))
```

`gradcheck` 用有限差分与解析梯度比较。默认容差面向 double precision；在不可微点、float32、随机运算或共享/重叠存储输入上，需要格外小心。官方说明见 [`gradcheck`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradcheck.html)。高阶导数可以用 `gradgradcheck` 检查。

## 14. 前向模式 AD 与反向模式 AD

虽然普通 `backward()` 主要是反向模式，PyTorch 也提供前向模式 AD。

| 模式 | 传播对象 | 高效的典型形状 |
| --- | --- | --- |
| 反向模式 | 从输出方向向输入传播 VJP：`vᵀJ` | 输入很多、输出很少，例如大量参数到标量 loss |
| 前向模式 | 随前向传播 JVP：`Jv` | 输入很少、输出很多 |

现代接口通常优先考虑：

```python
from torch.func import jvp, vjp, jacrev, jacfwd
```

例如 `torch.func.jvp(f, primals, tangents)` 返回函数值和 `Jv`。底层 forward AD 使用 primal 与 tangent 构成 dual tensor。当前官方仍提示前向 AD 的算子覆盖可能不完整，遇到“不支持某算子”需要查看对应 PyTorch 版本文档。

## 15. 不可微点、数值异常和复数梯度

### 15.1 不可微点不意味着一定报错

`relu(0)`、`abs(0)` 等点没有唯一普通导数。PyTorch 为各基础算子定义具体反向规则；官方总体原则包括优先使用正常导数、凸函数的最小范数次梯度、连续延拓等，但最终行为应以具体算子定义为准。

因此，不应假设“数学上不可微就一定异常”，也不应假设所有框架在不可微点都选同一个值。

### 15.2 前向屏蔽无效结果，未必能修复 backward

如果先计算除零，再用 mask 丢掉无效结果，危险运算可能已经进入图，反向仍可能得到 `nan`。更安全的思路是在执行危险运算**之前**构造有效输入或只选择有效元素。

### 15.3 复数梯度

对实值损失和复数参数，PyTorch 使用适合梯度下降的 conjugate Wirtinger derivative 约定。它与只看实数导数的直觉不同；编写复数自定义梯度时应专门阅读官方 [Autograd for Complex Numbers](https://docs.pytorch.org/docs/stable/notes/autograd.html#autograd-for-complex-numbers) 章节。

## 16. hooks：在梯度流经时观察或修改

常用 hook 有：

```python
handle = tensor.register_hook(fn)
```

`fn` 在该 Tensor 的梯度被计算时接收梯度，可用于监控、记录或返回替换后的梯度。叶子张量还可以使用 post-accumulate-grad hook，在 `.grad` 已累积后观察。

注意：

- hook 会进入训练执行路径，修改返回值会真正改变梯度；
- 注册顺序会影响多个 hook 的执行顺序；
- Tensor hook、Node pre/post hook、module full backward hook 的时机并不相同；
- 原地修改前后注册 hook 的归属可能令人意外。

因此，hook 更适合有明确生命周期管理的调试或算法功能；使用结束后可通过返回的 handle 移除。

## 17. 常见报错与定位思路

### 17.1 `element 0 of tensors does not require grad`

检查：

1. 目标 Tensor 的 `requires_grad` 是否为 `True`。
2. 中途是否调用 `.detach()`、`.item()`，或把数据转到 NumPy 后又重新构造 Tensor。
3. 前向是否处于 `no_grad` / `inference_mode`。
4. 是否使用了不支持所需梯度的操作或数据类型。

不要用“给最终 loss 强行设置 `requires_grad=True`”掩盖断图；那只会创建一个新的叶子，无法恢复此前已丢失的依赖关系。

### 17.2 `backward through the graph a second time`

检查是否对同一个前向结果反传了两次。普通训练应重新前向；确需同图多次 VJP 时，第一次反向使用 `retain_graph=True`，并评估内存成本。

### 17.3 `.grad` 是 `None`

依次判断：

- 它是不是非叶子张量；若是，是否提前 `retain_grad()`。
- 它是否真的参与了 loss 的计算。
- 是否被冻结为 `requires_grad=False`。
- 是否在反向前查看。
- optimizer 是否刚以 `set_to_none=True` 清理过梯度。

### 17.4 原地操作版本错误

开启 anomaly detection 可得到更接近出问题前向算子的 traceback：

```python
with torch.autograd.detect_anomaly():
    loss = model(x)
    loss.backward()
```

它还会检查 backward 产生的 `nan`，但开销明显，只应调试时使用。然后排查报错张量相关的 `_` 操作、切片赋值、view 别名和 `.data`。

### 17.5 梯度爆炸、消失或出现 `nan`

可按以下顺序定位：

1. 检查 loss 和激活第一次出现非有限值的位置。
2. 使用 `torch.isfinite` 检查参数与梯度。
3. 临时给关键 Tensor 或参数注册 hook，打印 norm、max、是否 finite。
4. 检查学习率、初始化、归一化、指数/对数/除法的定义域。
5. 必要时使用 anomaly detection 和 profiler；定位完应关闭调试开销。

## 18. 完整小例子推演

```python
import torch

x = torch.tensor([2.0, -1.0], requires_grad=True)
w = torch.tensor([3.0, 4.0], requires_grad=True)

a = x * w
a.retain_grad()
b = a.square()
loss = b.sum()

loss.backward()

print(a.grad)  # tensor([12., -8.])
print(x.grad)  # tensor([ 36., -32.])
print(w.grad)  # tensor([24.,  8.])
```

按执行顺序理解：

1. 创建 `x`、`w`。它们是需要梯度的叶子，尚无 `.grad`。
2. 执行乘法得到 `a`。因为输入需要梯度，`a` 是带 `grad_fn` 的非叶子。
3. `a.retain_grad()` 只要求反向后额外保存 `dL/da`，不改变数学计算。
4. 平方得到 `b`，求和得到单元素 `loss`。
5. `loss.backward()` 从种子梯度 `1` 开始。
6. `sum` 的局部 backward 把 `1` 扩展给 `b` 的每个元素：`dL/db=[1,1]`。
7. 平方的 backward 计算 `dL/da=(dL/db)⊙2a=[12,-8]`。
8. 乘法的 backward 分别计算：
   - `dL/dx=(dL/da)⊙w=[36,-32]`
   - `dL/dw=(dL/da)⊙x=[24,8]`
9. `x.grad`、`w.grad` 被填充；`a.grad` 因 `retain_grad()` 被额外保留。
10. 此图的反向保存状态默认释放；下一次训练迭代重新前向建新图。

再看非标量输出：

```python
x = torch.tensor([2.0, -1.0], requires_grad=True)
y = x.square()
v = torch.tensor([1.0, 10.0])
y.backward(v)
print(x.grad)  # tensor([4., -20.])
```

因为：

$$
J=\begin{bmatrix}2x_1&0\\0&2x_2\end{bmatrix}
=\begin{bmatrix}4&0\\0&-2\end{bmatrix}
$$

所以：

$$
v^TJ=[1,10]
\begin{bmatrix}4&0\\0&-2\end{bmatrix}
=[4,-20]
$$

这正是 `backward(v)` 的含义。

## 19. 对训练循环的最终理解

一个标准训练循环：

```python
for batch in loader:
    optimizer.zero_grad(set_to_none=True)
    prediction = model(batch.x)
    loss = criterion(prediction, batch.y)
    loss.backward()
    optimizer.step()
```

逐行对应的机制是：

1. `zero_grad`：清理上一次累积在参数叶子上的梯度。
2. `model(...)`：执行普通 Tensor 运算，并为本次实际路径建立动态图。
3. `criterion(...)`：把模型输出继续接到标量 loss，图也继续延伸。
4. `loss.backward()`：从 `1` 开始执行反向图的局部 VJP，合并分支贡献，填入参数 `.grad`。
5. `optimizer.step()`：读取 `.grad`，在不把更新纳入当前图的情况下修改参数。
6. 下一轮使用更新后的参数重新前向，建立全新的图。

这里最值得牢牢记住的是：

- autograd 求的是当前执行路径的导数，不是静态分析整份 Python 程序。
- backward 传播的是 VJP；标量 loss 只是把种子梯度简化成了 `1`。
- 中间节点梯度会被计算，但默认只把叶子梯度保存在 `.grad`。
- `.grad` 会累积，不会每轮自动清空。
- 反向所需的前向数据默认在使用后释放；`retain_graph` 不是常规训练开关。
- `create_graph` 才是让梯度计算本身可继续求导的开关。
- `eval()` 与禁用梯度是两个独立维度。
- 原地操作的风险来自覆盖 backward 所需的数据；版本计数负责尽早发现这类错误。
- optimizer 更新参数，autograd 只负责计算梯度。

## 20. 自查练习与验证命令

### 20.1 建议手算

1. 对 `y = x*x + 3*x`，在 `x=2` 时手算 `dy/dx`，再解释两条路径怎样相加。
2. 对 `y = x**2`、`x=[1,2,3]`，分别传入 `v=[1,0,0]` 和 `v=[1,1,1]`，预测两次 VJP。
3. 对形状 `(B,D)` 的 `x` 和 `(D,)` 的 bias，推导 `sum(x+b)` 对 bias 的梯度为何包含 batch 维求和。
4. 解释为什么 `model.eval()` 后仍然可以调用 `loss.backward()`。
5. 解释 `detach()` 与 `clone()` 分别改变 autograd 历史和存储关系中的哪一部分。

### 20.2 在本项目环境运行示例

```bash
uv run python
```

然后逐段粘贴第 18 节示例，观察：

```python
print(x.is_leaf, x.requires_grad, x.grad_fn)
print(a.is_leaf, a.requires_grad, a.grad_fn)
print(type(loss.grad_fn).__name__)
```

### 20.3 本次实际验证

本次在本项目 `torch 2.11.0` CPU 环境运行了第 18 节的叶子/非叶子、VJP 和二阶导数示例，得到：

```text
a.grad = [12, -8]
x.grad = [36, -32]
w.grad = [24, 8]
非标量例子的 VJP = [4, -20]
x=2 时，x³ 的一阶导数 = 12，二阶导数 = 12
```

这些结果与手算一致。本文没有进行 GPU、复数梯度、自定义 CUDA 算子或分布式 autograd 的运行验证。

## 参考资料

- [PyTorch Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)：动态图、saved tensors、不可微点、grad modes、原地操作、复数梯度和 hooks。
- [Automatic differentiation package - `torch.autograd`](https://docs.pytorch.org/docs/stable/autograd.html)：API 总览、高阶接口、anomaly detection 和数值梯度检查。
- [`torch.Tensor.backward`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html)：`gradient`、`retain_graph`、`create_graph` 和 `inputs` 的正式语义。
- [Automatic Differentiation with `torch.autograd`](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)：基础计算图、梯度累积和 VJP 示例。
- [Understanding requires_grad, retain_grad, Leaf, and Non-leaf Tensors](https://docs.pytorch.org/tutorials/beginner/understanding_leaf_vs_nonleaf_tutorial.html)：叶子、非叶子和 `retain_grad()`。
- [Extending PyTorch](https://docs.pytorch.org/docs/stable/notes/extending.html)：自定义 `torch.autograd.Function`、`save_for_backward`、`backward` 与 `jvp`。
- [`torch.func.jvp`](https://docs.pytorch.org/docs/stable/generated/torch.func.jvp.html)：前向模式 JVP 接口。
- [`gradcheck`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradcheck.html)：有限差分梯度校验及其限制。
- [Overview of PyTorch Autograd Engine](https://pytorch.org/blog/overview-of-pytorch-autograd-engine/)：反向引擎的 Node、Edge、依赖与调度概念；文章基于较早版本，适合辅助理解，不应视为当前内部实现的稳定 API 契约。
