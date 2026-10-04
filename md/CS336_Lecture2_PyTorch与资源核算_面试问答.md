# CS336 Lecture 2：PyTorch 与资源核算（知识点及面试问答）

整理日期：2026-10-03  
对应课程：Stanford CS336 Spring 2026, Lecture 2：PyTorch（einops）、FLOPs、内存、算术强度与训练资源优化。  
主要依据：[课程页面](https://cs336.stanford.edu/lectures/?trace=lecture_02)、[官方讲义源码 `lecture_02.py`](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)。

本文中的“面试题”是根据讲义知识点**自行设计的模拟问答**，不是对真实面经的统计。公式除非另有说明，都是忽略常数级开销的估算；实际训练应以测量为准。这里讨论的是 2026 版 Lecture 2，不能直接套用旧版讲义的代码和数值。

## 一、先建立整节课的框架

这节课的核心问题是：给定一段模型计算，如何估计它需要多少内存、多少浮点运算，以及运行时更可能受计算速度还是内存带宽限制。

```text
张量形状、dtype、设备
    ├─ 元素数 × 每元素字节数 → 张量数据量
    ├─ 操作类型与维度       → FLOPs
    └─ FLOPs / 搬运字节数   → 算术强度
                               ↓
              与硬件计算/带宽之比比较 → 瓶颈与性能上界
                               ↓
       训练还要加上反向传播、梯度、优化器状态和激活值
                               ↓
              梯度累积、激活检查点等内存优化
```

建议记住四个单位：`byte` 表示数据量；`FLOP` 表示运算量；`FLOP/s` 表示计算吞吐率；`byte/s` 表示内存带宽。`FLOPs` 与 `FLOP/s` 在口语中常混用，面试时最好主动写明单位。

## 二、张量、内存与精度：重点展开

### 1. `get_memory_usage` 如何使用？

讲义末尾定义的辅助函数是：

```python
def get_memory_usage(x: torch.Tensor):
    return x.numel() * x.element_size()
```

- 参数 `x`：需要估算数据量的 PyTorch 张量。
- `x.numel()`：张量的**逻辑元素个数**，例如形状 `(4, 8)` 有 `32` 个元素。
- `x.element_size()`：每个元素的字节数，例如 `float32` 是 `4` 字节，`bfloat16` 是 `2` 字节。
- 返回值：`元素数 × 每元素字节数`，单位为 **byte**，不是 MB，也不是整个进程或 GPU 的峰值显存。

按执行顺序看：先由 `numel()` 得到元素数，再由 `element_size()` 得到单元素字节数，最后相乘。调用示例：

```python
x = torch.zeros(4, 8, dtype=torch.float32)
get_memory_usage(x)             # 32 × 4 = 128 bytes
get_memory_usage(x) / 2**20     # 换算为 MiB
```

**最容易混淆的地方：**这个函数只计算张量中逻辑元素对应的数据量。视图可能与原张量共享底层存储；内存分配器会有缓存和碎片；训练过程中还有其他张量和临时工作空间。因此不能把它的结果当作程序的真实显存占用，也不能简单相加所有视图。若要测 CUDA 运行过程，可参考讲义中的 `get_max_memory_usage`，它调用 `torch.cuda.max_memory_allocated()`；该值也有自己的统计范围，不等于 `nvidia-smi` 显示的全部显存占用。参见 [官方讲义](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)、[PyTorch `element_size` 文档](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.element_size.html) 和 [CUDA 内存诊断文档](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)。

讲义中的大矩阵例子：形状 `(12288 × 4, 12288)`、`float32`，数据量为 `49,152 × 12,288 × 4 = 2,415,919,104 bytes = 2.25 GiB`。注意 `GB`（十进制）与 `GiB`（二进制）的换算不同。

### 2. 精度选择为什么会影响训练？

| 类型 | 字节/元素 | 指数位 | 尾数位 | 主要特点 |
| --- | ---: | ---: | ---: | --- |
| fp32 | 4 | 8 | 23 | 范围与精度都较高，内存和带宽成本较大 |
| fp16 | 2 | 5 | 10 | 比 fp32 省一半数据量，但动态范围较窄，小梯度可能下溢 |
| bf16 | 2 | 8 | 7 | 动态范围接近 fp32，但有效精度比 fp16 更粗 |
| fp8 | 1 | 随格式变化 | 随格式变化 | 依赖具体格式、缩放策略和硬件支持 |
| fp4 | 0.5（原始数值位数） | 依格式而定 | 依格式而定 | 常需块级缩放等元数据，实际占用并非严格 0.5 字节/值 |

讲义用 `1e-8` 举例：转换成 fp16 可能变为 0，而 bf16 仍可表示非零值。其原因是 bf16 保留了与 fp32 相同宽度的指数，而不是它“处处比 fp16 更精确”。训练中常用混合精度，让适合低精度的计算获得吞吐和内存收益，同时让敏感计算或长期累积状态保留更高精度。

**AMP 的精确表述：**`torch.amp.autocast` 会对符合条件的算子选择执行 dtype；它**不会**自动把模型所有参数或 `torch.zeros(...)` 创建的张量永久改成 bf16。讲义在 autocast 区域里调用 `torch.zeros(4, 8)`，不能据此推断 `x.dtype` 已变成 bf16。理解混合精度账本时，应分别核实参数、激活、梯度、优化器状态和可能存在的 master weights 的真实 dtype。参见 [PyTorch AMP 文档](https://docs.pytorch.org/docs/stable/amp.html)。

### 3. 张量的设备和形状

张量的 `rank` 是维度个数。例如 Transformer 常见的 `(B, S, H, D)` 是四维，其中 `B` 为 batch size、`S` 为序列长度、`H` 为注意力头数、`D` 为每头维度。形状决定元素数，dtype 决定每元素字节数。默认创建的张量在 CPU；`x.to(device)` 会得到目标设备上的张量，或直接在创建时指定 `device=`。多张量计算必须留意设备与 dtype 是否匹配。

## 三、einops：用轴名表达张量操作

讲义把 `einsum`、`reduce` 和 `rearrange` 作为三类常用操作。轴名的好处是能直接表达“哪一维参与运算、哪一维保留”，从而减少 `transpose(-2, -1)` 一类位置编号错误。

```python
# x: [batch, seq1, hidden], y: [batch, seq2, hidden]
scores = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")

# x: [batch, seq, hidden]
summed = reduce(x, "batch seq hidden -> batch seq", "sum")

# x: [seq, heads * head_dim]
split = rearrange(x, "seq (heads head_dim) -> seq heads head_dim", heads=2)
```

第一式输出 `[batch, seq1, seq2]`；`hidden` 在输入出现、输出消失，所以沿该轴求和；`batch` 保留，逐 batch 计算。第二式沿 `hidden` 求和。第三式只重排维度，要求合并维长度能被已知的 `heads` 整除。这里用的是 **`einops.einsum`** 的完整轴名语法，和 `torch.einsum` 的单字母字符串写法不同。参见 [einops 官方文档](https://einops.rocks/api/einsum/)。

注意：形状相同不代表语义相同。例如 Q、K 的 `seq1` 和 `seq2` 可以长度相等，但在 attention score 中分别对应查询位置和键位置。先写出每个张量的轴含义，再写公式，比先试 `transpose` 更可靠。

## 四、FLOPs、反向传播与训练时长：重点展开

### 1. 矩阵乘法怎么数 FLOPs？

设 `X` 的形状为 `(B, D)`、`W` 为 `(D, K)`，则 `Y = XW` 为 `(B, K)`。每个输出元素是长度为 `D` 的点积，约有 `D` 次乘法和 `D` 次加法；共有 `BK` 个输出，因此：

\[
F_{\mathrm{forward}} \approx 2BDK.
\]

更精确的朴素计数是 `BK(2D−1)`：每个点积有 `D` 次乘法、`D−1` 次加法。工程估算通常取 `2BDK`，并默认一次乘加算 **2 FLOPs**。

### 2. 为什么一层线性的反向传播约为前向的两倍？

设上游梯度 `G = ∂L/∂Y`，形状 `(B, K)`。反向需算两次矩阵乘法：

\[
\frac{\partial L}{\partial W}=X^\top G \quad(D,K),
\qquad
\frac{\partial L}{\partial X}=GW^\top \quad(B,D).
\]

两项各约 `2BDK` FLOPs，反向合计约 `4BDK`；再加前向 `2BDK`，总计约 `6BDK`。首层若输入无需梯度，可省去对应的 `∂L/∂X`，所以“反向恰好两倍”是典型层的近似，不是每个算子的定律。

若模型主要由大规模线性层构成，参数总量为 `P`，处理 `N` 个 token，可用 `训练 FLOPs ≈ 6PN` 做**第一轮**估算。讲义明确说这是 MLP 推导，对较短上下文的 Transformer 也常有参考价值；长上下文时 attention 的二次项、softmax、归一化等不能忽略。这里的 `N` 是**整个训练过程处理的 token 总数**，不是一个 step 的 batch size。

### 3. 讲义中的训练时长与模型容量估算

以讲义设定的 `70B` 参数、`15T` token 为例：

\[
6PN = 6\times70\times10^9\times15\times10^{12}
= 6.3\times10^{24}\ \mathrm{FLOPs}.
\]

若假设 `1024` 张 H100、每张密集计算峰值约 `989.5 TFLOP/s`、利用率 `50%`，则理论估计约 **144 天**。这个结果只是讲义假设下的算术估算；数据加载、通信、评估、故障、额外计算和峰值口径变化都可能改变实际时间。

讲义另一道题把每张 H100 的 `80 GB`、8 张卡、每参数 `12 bytes` 代入，得到 `8×80 GB÷12 ≈ 53.3B` 参数。这是**假设模型状态能跨 8 卡合理切分、且完全忽略激活等开销的容量上界**；不能理解为 53.3B 模型可以在每张卡上完整复制训练。

### 4. MFU 与计时

讲义用 `MFU = 模型 FLOPs / (运行时间 × 硬件理论峰值 FLOP/s)` 估计模型计算利用率。比较时需要固定硬件、dtype、稠密/稀疏口径、模型 FLOPs 的统计范围。GPU 操作异步执行，因此只在 Python 中直接计时，可能只测到 kernel 提交时间；讲义的 `benchmark` 会在计时前后调用 `torch.cuda.synchronize()`。理论峰值是规格书数字，真实吞吐还受 kernel、数据形状、通信和调度影响。

## 五、算术强度与 Roofline：重点展开

算术强度 `I = FLOPs / HBM 搬运字节数`，单位为 FLOP/byte。硬件平衡点 `I* = 峰值计算吞吐率 / HBM 带宽`，单位也为 FLOP/byte。理想化的 roofline 上界为：

\[
\mathrm{throughput} \le \min(F_{\mathrm{peak}},\; I\,B_{\mathrm{peak}}).
\]

当 `I < I*`，内存带宽上界更低，属于带宽受限；当 `I > I*`，计算峰值上界更低，属于计算受限。按讲义中的 H100 示例值，`I* ≈ (989.5 TFLOP/s)/(3.35 TB/s) ≈ 295 FLOP/byte`。数字只对应讲义选择的 H100、dtype 和峰值口径。

| 操作（bf16） | 主要搬运量（理想化） | 主要工作量 | 算术强度约值 | 讲义中的结论 |
| --- | --- | --- | --- | --- |
| 长度 `n` 的 ReLU | 读 `2n`、写 `2n` bytes | 约 `n` 个逐元素操作 | `1/4` 操作/byte | 带宽受限 |
| 长度 `n` 的点积 | 读两个向量约 `4n` bytes | `2n` FLOPs | `1/2` | 带宽受限 |
| `n×n` 矩阵乘向量 | 主要读矩阵约 `2n²` bytes | `2n²` FLOPs | `1` | 带宽受限 |
| 两个 `n×n` 矩阵相乘 | 读两矩阵、写结果约 `6n²` bytes | `2n³` FLOPs | `n/3` | 足够大时可计算受限 |

例如 `n=1024` 的方阵乘法，理想算术强度约 `341 FLOP/byte`，高于上面的 H100 平衡点。原因是矩阵块可以复用，同一份从 HBM 读取的数据参与许多次乘加。向量乘矩阵时，巨大的权重矩阵通常要读取一次，却只贡献约等量级的 FLOPs，因此低 batch 的自回归 **decode** 往往更容易受带宽限制；大 batch 或 **prefill** 则可能表现不同。

以上表格是假定读写次数、缓存复用、算子实现都较理想的模型。ReLU 中的“比较”并不是严格意义上的浮点加/乘 FLOP，这一行是讲义为了分析瓶颈而采用的工作量近似。Roofline 只给性能**上界**，不能写成“MFU 必然等于 `min(1, I/I*)`”；实际 MFU 往往更低。GELU 比 ReLU 计算更多，但在同样搬运量下仍可能落在带宽受限区间，不能只凭操作数断言它一定慢多少。参见 [讲义的 Roofline 部分](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py) 和 [Roofline 解释](https://jax-ml.github.io/scaling-book/roofline/)。

## 六、训练状态、优化器和内存优化

### 1. 训练时显存分成哪些部分？

| 部分 | 内容 | 主要随什么增长 |
| --- | --- | --- |
| 参数 | 权重与偏置 | 参数量 `P`、参数 dtype |
| 梯度 | 每个可训练参数的梯度 | `P`、梯度 dtype |
| 优化器状态 | 如 Adam 的一阶、二阶矩 | `P`、状态 dtype、优化器种类 |
| 激活值 | 反向传播可能需要的前向中间量 | batch、序列长度、层数、隐藏维度；attention 实现也有影响 |
| 临时缓冲 | 算子 workspace、通信缓冲、分配器开销等 | 实现与运行环境 |

在讲义的**假设场景**中，参数 bf16 为 `2P` bytes、梯度 bf16 为 `2P`、Adam 两个 fp32 矩为 `8P`，合计 `12P` bytes，**尚未计激活和其他缓冲**。若参数和梯度也为 fp32，则同样账本是 `4P+4P+8P=16P` bytes。若额外保存 fp32 master weights，还要再加相应空间。不能把 `12P` 当成所有 PyTorch AdamW 训练的固定事实：状态实际 dtype 要看所用优化器实现和精度策略。

讲义的 `AdaGrad` 示例解释了优化器状态：它累计平方梯度 `g2`，用 `grad / sqrt(g2 + eps)` 缩放更新；若 `g2` 是 fp32，约需每参数额外 4 字节。Adam 保存两个矩，则通常按 8 字节估算。**源码细节提醒：**讲义手写的 `AdaGrad` 用 `torch.zeros_like(grad)` 初始化 `g2`，其 dtype 会跟随 `grad`；旁边的“fp32 状态”内存表是另设的估算假设，不能直接说这段示例代码已将状态转成 fp32。

训练循环的顺序是：取 batch → 前向求 loss → `loss.backward()` → `optimizer.step()` → `optimizer.zero_grad(set_to_none=True)`。PyTorch 默认会将多次 `backward()` 的梯度**累积**到 `.grad`，这既是忘记清零时的常见错误，也是梯度累积的基础。`nn.Parameter` 和 `nn.ModuleList` 用于让权重及子层被模块正确注册。参见 [官方讲义](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)。

### 2. 梯度累积

目标有效 batch 为 `B`，每次只能放下微批 `b`，则做 `k=B/b` 次前向与反向，在第 `k` 次之后才更新参数。若每个微批的 loss 已按自身样本求平均，通常将其除以 `k` 再 `backward()`，才能与一个大小为 `B` 的 batch 的**平均梯度**对应；最后不满批和按 token 加权的 loss 要按实际样本/token 数处理。

梯度累积主要降低**单次驻留激活值**，不会自动缩小参数、梯度和优化器状态。它可以获得近似相同的有效 batch 梯度，但微批之间的 BatchNorm 等跨样本操作、随机性和更新频率可能使行为不同；吞吐也未必不变。

### 3. 激活检查点（activation checkpointing）

正常反向传播需要保存一些前向中间量。检查点策略只保存部分边界激活，反向时从最近边界重新计算内部激活，以额外计算换更低峰值内存。课程用“每隔约 `√L` 层保存一次”说明经典分段模型：边界和段内重算的峰值都约 `O(√L)`，总重算量仍为 `O(L)` 量级。这里的复杂度只针对简化的、等宽等成本的层链；实际内存还有参数、梯度和工作空间。

讲义代码 `DeepNetworkCheckpointed.forward()` 是**逐层**调用 `torch.utils.checkpoint.checkpoint(layer, x)` 的演示；“每隔 `√L` 层”是随后讨论的策略，不是这段代码已经实现的设置。现代 PyTorch 文档建议显式传 `use_reentrant=False`，实际使用时还需注意随机数状态与函数重算的一致性。参见 [PyTorch checkpoint 文档](https://docs.pytorch.org/docs/stable/checkpoint.html)。

## 七、AI 面试高频模拟 Q&A

以下回答刻意控制在面试口述长度；遇到追问时，可回到前面的推导。

### A. 张量与精度

**Q1：给一个形状 `(B, S, D)` 的 bf16 激活，如何估算其数据量？**  
A：逻辑元素数是 `B×S×D`，bf16 为 2 字节/元素，所以约 `2BSD` bytes。若问训练峰值，还需知道保存了哪些层、是否有额外 attention 矩阵和临时缓冲。

**Q2：`get_memory_usage(x)` 与 `torch.cuda.max_memory_allocated()` 有何区别？**  
A：前者是单张量的逻辑数据量 `numel×element_size`；后者统计 PyTorch CUDA 分配器在一段运行期间的峰值已分配显存。两者都不等价于整个 GPU 的占用。

**Q3：为什么 bf16 与 fp16 都是 2 字节，训练表现却可能不同？**  
A：bf16 指数位更多、动态范围接近 fp32，因此更能避免范围不足导致的上溢/下溢；代价是尾数更少，近似值更粗。不能简单说“bf16 在所有数值上更准确”。

**Q4：用了 autocast，模型参数就变成 bf16 了吗？**  
A：不一定。autocast 主要为符合条件的算子选择执行精度，参数存储 dtype 取决于模型如何创建或转换；要分别检查 `param.dtype`、`param.grad.dtype` 和优化器状态。

**Q5：为什么 8 位或 4 位训练的显存不能只用 `P×1` 或 `P×0.5` 估算？**  
A：缩放因子、元数据、更高精度累积、梯度和优化器状态都可能额外占用；低位表示也不保证所有训练算子都能原生用该精度执行。

### B. 形状、FLOPs 与自动求导

**Q6：`einsum("batch query hidden, batch key hidden -> batch query key")` 做了什么？**  
A：对同一 batch 内每个 query/key 向量做 hidden 维点积，生成 `[batch, query, key]` 分数矩阵；`hidden` 未出现在输出端，故被求和。

**Q7：`rearrange` 拆多头维度时最常见的形状错误是什么？**  
A：总隐藏维 `D_model` 必须满足 `D_model = H×D_head`；拆分后要确认头轴位置及后续矩阵乘的轴含义。形状数值碰巧相同也可能掩盖轴顺序错误。

**Q8：为什么 `(B,D) @ (D,K)` 约为 `2BDK` FLOPs？**  
A：有 `BK` 个输出，每个输出做长度 `D` 的乘加，按乘法和加法各一次计约 `2D`，因此约 `2BDK`。

**Q9：如何手推 `Y=XW` 的反向计算量？**  
A：`dW=XᵀdY` 与 `dX=dYWᵀ` 是两次与前向同阶的矩阵乘，各约 `2BDK`；合计反向约 `4BDK`，加前向约 `6BDK`。

**Q10：为什么不能把 `6PN` 当成任何 Transformer 的精确 FLOPs？**  
A：它主要近似大矩阵乘的前后向成本，未完整覆盖 attention 的 `S²` 项、softmax、归一化、输出投影细节、重算等。序列越长，漏掉的项越可能显著。

**Q11：`loss.backward()` 后再次调用 `loss.backward()` 会怎样？**  
A：对不同前向图连续调用时，参数 `.grad` 默认相加；对同一个已释放的计算图重复调用通常会报错，除非保留图。梯度累积需要多个微批各自前向、按适当比例计算 loss，并在更新后清梯度。

### C. Roofline 与性能

**Q12：算术强度和硬件平衡点分别是什么？**  
A：算术强度是工作量/数据搬运量，单位 FLOP/byte；硬件平衡点是峰值 FLOP/s 除以峰值 byte/s。前者低于后者时，带宽形成更低的性能上界。

**Q13：为什么矩阵乘法常比矩阵向量乘法更能用满计算单元？**  
A：矩阵乘法可在多个输出上复用已读入的矩阵块，FLOPs 随 `n³` 增长而理想数据搬运约随 `n²` 增长；矩阵向量乘法的大矩阵读入量与 FLOPs 都约随 `n²` 增长，复用较少。

**Q14：低 batch 的 LLM decode 为什么常受内存带宽限制？**  
A：生成一个新 token 时，大量权重需要读取，但每个权重参与的计算次数少，类似矩阵向量乘法。增大 batch 可以增加权重复用；prefill 的大矩阵乘法则可能转为计算受限。还需考虑 KV cache 和 kernel 实现。

**Q15：GELU FLOPs 多于 ReLU，就一定明显更慢吗？**  
A：不能只看 FLOPs。如果两者都受读写激活的带宽限制，额外算术不一定主导总耗时；实际还受算子融合、近似实现和 kernel 开销影响，应测量确认。

**Q16：Roofline 图能直接预测实际 MFU 吗？**  
A：不能。它给出 `min(计算峰值, 算术强度×带宽)` 的理想上界；实际可能受缓存、启动开销、占用率、算子形状、通信和实现限制。

**Q17：为什么 GPU benchmark 要同步？**  
A：CUDA kernel 通常异步提交；如果不在计时边界同步，计时可能只覆盖 CPU 的提交时间，而没有等 GPU 完成。

### D. 训练显存与优化

**Q18：AdamW 为什么常见“每参数 12 字节”的说法？**  
A：在“bf16 参数 2 + bf16 梯度 2 + fp32 一阶矩 4 + fp32 二阶矩 4”这个明确假设下合计 12 字节。换 dtype、保留 master weights 或增加临时缓冲，数字就变了。

**Q19：8 张 80 GB GPU 就能直接训练 53B 参数模型吗？**  
A：`8×80 GB÷12≈53B` 只是把全部显存都用于模型状态、且跨卡切分均匀时的粗略上界。真实训练还需为激活、通信、工作空间等留容量；若每卡复制完整状态，这个除以 8 的算法不成立。

**Q20：为什么单纯减小 batch 可能仍然 OOM？**  
A：减小 batch 主要减少激活相关占用，参数、梯度和优化器状态基本不变；如果固定状态已经逼近显存上限，需要状态切分、降低精度、换优化器等其他办法。

**Q21：梯度累积能否完全等价于一个大 batch？**  
A：对可加的逐样本损失，正确处理每个微批的权重后，得到的梯度可接近大 batch；但跨样本操作、随机状态、浮点求和顺序和优化器更新时机可能造成差异。

**Q22：为什么微批平均 loss 要除以累积次数？**  
A：若每次 `backward()` 得到的是各微批的平均梯度，直接相加会放大 `k` 倍；除以 `k` 后才对应等大小微批合并后的整体平均梯度。

**Q23：activation checkpointing 是如何以计算换内存的？**  
A：前向只保留检查点边界，反向计算梯度时重跑部分前向以恢复内部激活，因此少存激活、多做计算。

**Q24：梯度累积与激活检查点解决的是同一个问题吗？**  
A：都能降低激活峰值，但机制不同：梯度累积缩小单次微批；检查点减少一个微批内保存的中间量。两者可组合，且都不会凭空消除参数和优化器状态。

**Q25：如果理论算出的模型显存小于 GPU 容量却仍 OOM，你会先检查什么？**  
A：核对参数、梯度、状态和激活的真实 dtype/形状；查是否有 attention 中间矩阵、保存的计算图、临时工作空间和分配器缓存；用峰值显存统计或 profiler 定位 OOM 发生的具体步骤。

## 八、三道手算自测

1. `x.shape=(32, 2048, 4096)` 且为 bf16。只算 `x` 的逻辑数据量：`32×2048×4096×2 = 536,870,912 bytes = 512 MiB`。
2. `X:(128, 4096)`、`W:(4096, 4096)`。前向矩阵乘约 `2×128×4096² ≈ 4.29×10⁹ FLOPs`；若输入也需梯度，前后向总量约为前向的 3 倍。
3. 有 `L=36` 个等宽层。按课堂简化的分段模型，每约 `√36=6` 层设检查点，可把边界数量和段内需暂存的激活数量都降到约 6 的量级；这不是精确显存数字。

复习时先问自己：**形状是什么？每个元素多少字节？数据需要读写几次？一次前向之外还要保存什么、重算什么？** 把这些问题写清楚，再套公式通常不会偏得太离谱。

## 参考资料

- [Stanford CS336 Spring 2026 Lecture 2 页面](https://cs336.stanford.edu/lectures/?trace=lecture_02)
- [Stanford CS336 官方 `lecture_02.py`](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)
- [Stanford CS336 官方 `gpu_util.py`](https://github.com/stanford-cs336/lectures/blob/main/gpu_util.py)
- [PyTorch AMP 文档](https://docs.pytorch.org/docs/stable/amp.html)
- [PyTorch activation checkpointing 文档](https://docs.pytorch.org/docs/stable/checkpoint.html)
- [PyTorch CUDA 内存诊断文档](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)
- [einops `einsum` 文档](https://einops.rocks/api/einsum/)
- [The Scaling Book：Roofline](https://jax-ml.github.io/scaling-book/roofline/)
