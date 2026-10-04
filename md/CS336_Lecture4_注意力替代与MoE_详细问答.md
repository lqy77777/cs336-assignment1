# CS336 Lecture 4：注意力替代方案与混合专家模型（详细问答）

整理日期：2026-10-03  
来源：用户提供的 60 页 [lecture_04.pdf](/Users/lambert7/Desktop/lecture_04.pdf)（CS336 Spring 2026，Tatsu Hashimoto）；可对照[课程官方 PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_04.pdf)。  
范围：线性注意力及循环形式、Mamba-2/Gated DeltaNet、混合与稀疏注意力、MoE 路由与专家设计、负载均衡、系统实现、训练稳定性、upcycling，以及 DeepSeek-V1/V2/V3 的 MLA/MTP。本文问题是**根据讲义设计的复习和模拟面试题，不是真实面经**。

> 阅读约定：页码按这份 60 页 PDF 标注。每题先给适合口述的结论，再解释前提、公式或常见追问。标为“课件案例”的模型对比只说明该实验条件下的观察；标为“推导”的复杂度和公式须满足所写假设。不要把架构示意图当作完整工程实现。

## 0. 一页路线图与统一符号

```text
长上下文成本
  ├─ 稠密 softmax 注意力：每个 query 看大量历史 token
  ├─ 线性/循环状态：压缩历史为固定尺寸状态，换取低成本但有表达取舍
  ├─ 混合层：用少数 full attention 补充精确检索能力
  └─ 动态稀疏注意力：先找候选，再只对候选做昂贵的注意力

模型容量成本
  └─ MoE：大量专家提供总容量；每个 token 只激活少数专家
       ├─ 算法：选谁、如何计算权重、如何防止塌缩
       └─ 系统：分发、跨卡通信、不均衡、容量与精度
```

记序列长度为 $n$，注意力 key/query 维度为 $d_k$、value 维度为 $d_v$；MoE 的 routed 专家数为 $N$，每 token 选 $k$ 个，shared 专家数为 $N_s$，batch 中 token 数为 $T$。$P_{\rm total}$ 指模型实际存放的全部参数，$P_{\rm active}$ 指处理某个 token 时用到的参数。二者不能互换；训练显存、推理显存、每 token FLOPs 和通信量也不能互换。

## 1. 为什么需要注意力替代方案？（PDF 第 1–3 页）

**Q1：这节课的两条主线是什么？**  
A：第一条是**长上下文效率**：减少 dense attention 对所有位置的配对计算或缓存读取。第二条是**模型容量效率**：用 MoE 让总参数量大于每 token 实际计算量。前者主要沿序列维度稀疏化，后者沿专家/参数维度稀疏化。

**Q2：标准自注意力为什么随上下文增长昂贵？**  
A：$QK^\top$ 对 $n$ 个 query 与 $n$ 个 key 做配对，矩阵有 $n^2$ 个元素，乘法约 $O(n^2d_k)$；再乘 $V$ 约 $O(n^2d_v)$。因果掩码使有效配对数约减半，但渐近复杂度仍是 $O(n^2)$。实际显存可通过 FlashAttention 等避免显式保存完整注意力矩阵，所以“数学上有 $n^2$ 配对”不等于“必须占 $n^2$ 激活显存”。

**Q3：prefill 与逐 token decode 在长上下文时的成本有何不同？**  
A：prefill 同时处理一段长度 $n$ 的上下文，dense attention 的配对计算随 $n^2$ 增长；decode 的每个新 token 通常只对已有 $n$ 个 key/value 做注意力，单步注意力约 $O(n)$，而 KV cache 与历史长度也约成正比。两个阶段的瓶颈不一定相同。

**Q4：已有的“基础工具箱”是什么？**  
A：课件第 3 页概括为局部与全局注意力混用、以及系统工程优化。局部窗口把每个 query 的候选限定为邻近 $w$ 个位置；全局层保留远程信息通道；系统手段优化内核、布局和访存。后面的线性/状态模型与动态稀疏方法是更激进的替代。

**Q5：比较长上下文方案应报哪些指标？**  
A：至少包括相同硬件与 batch 下的 prefill 吞吐/延迟、decode 吞吐/延迟、KV/状态显存、有效上下文长度、训练成本、验证 loss 和长程检索/推理任务表现。只报 FLOPs 或只报某个 benchmark，不能覆盖全部取舍。

**Q6：面试官问“长上下文只要把 full attention 换成 O(n) 就一定更好”，如何答？**  
A：渐近复杂度只是一个维度。还要看状态能否保留所需信息、并行训练算法是否高效、短序列常数项、长序列数据分布、以及实际 kernel/显存带宽。近似或压缩历史可能损失精确检索能力，因此常出现混合设计。

## 2. 线性注意力：结合律、循环状态与边界（PDF 第 4–5 页）

### 核心推导

若暂把注意力中的非线性 $\rho$ 取为恒等映射，且 $Q\in\mathbb R^{n\times d_k}$、$K\in\mathbb R^{n\times d_k}$、$V\in\mathbb R^{n\times d_v}$：

$$
(QK^\top)V=Q(K^\top V).
$$

左结合约需 $O(n^2d_k+n^2d_v)$ 乘加，右结合先求 $d_k\times d_v$ 的 $K^\top V$，再左乘 $Q$，约需 $O(2nd_kd_v)$。这仅是**无 softmax 的矩阵结合律**，不能直接声称原版 softmax attention 已变成线性。对因果模型，也不能一次用全序列的 $K^\top V$ 回答过去位置，否则会泄露未来；应使用前缀状态：

$$
S_t=S_{t-1}+k_tv_t^\top=\sum_{s\le t}k_sv_s^\top,\qquad y_t=q_t^\top S_t.
$$

其中 $S_t\in\mathbb R^{d_k\times d_v}$，$k_t,q_t\in\mathbb R^{d_k}$，$v_t,y_t\in\mathbb R^{d_v}$。这里每步更新/读取约 $O(d_kd_v)$，状态约 $d_kd_v$ 个元素，不随历史 token 数线性增长。

**Q7：线性注意力里的“线性”指什么？**  
A：通常指在固定特征/状态维度、固定层数下，沿序列长度 $n$ 的计算或状态更新成本为 $O(n)$，不是输出对输入是线性函数，也不是所有维度的总成本都是常数。

**Q8：为什么 softmax 阻止直接交换矩阵乘法顺序？**  
A：$\operatorname{softmax}(QK^\top)V$ 中 softmax 逐 query、跨 key 归一化，不能通过矩阵乘法结合律移到 $K^\top V$ 的右结合形式。把 $\rho$ 换成恒等函数会改变模型，不是等价优化。

**Q9：怎样把第 4 页的线性代数式改写成因果推理算法？**  
A：维护前缀状态 $S_t=\sum_{s\le t}k_sv_s^\top$，新 token 来时做外积更新 $k_tv_t^\top$，然后计算 $q_t^\top S_t$。因果性来自前缀上界 $s\le t$；若 $S_t$ 含未来 token，就产生信息泄露。

**Q10：$S_t$ 的每个元素有什么含义？**  
A：$S_t[a,b]=\sum_{s\le t}k_s[a]v_s[b]$，汇总了过去 key 第 $a$ 个特征与 value 第 $b$ 个特征的相关贡献。query 的第 $a$ 个分量再决定读取这一行的权重。

**Q11：能手算一个极小例子吗？**  
A：令 $d_k=d_v=1$，$k_1=2,v_1=3,k_2=1,v_2=4$。那么 $S_1=6$，$S_2=6+4=10$；若 $q_2=0.5$，则 $y_2=5$。普通 softmax attention 对这两个 value 会产生归一化权重的加权平均，结果一般不同。

**Q12：状态固定大小是否意味着可以无损记住无限历史？**  
A：不能。固定尺寸状态把越来越长的历史压到有限维摘要里，会有信息容量和干扰问题。除非任务本身只依赖可压缩统计量，否则不能保证精确恢复任意历史 token；这也是保留一些 full attention 的动机之一。

**Q13：纯线性式没有 softmax，输出会出现什么数值或建模问题？**  
A：$S_t$ 的幅度可能随长度累积，输出不再是非负且和为 1 的权重形成的凸组合，也可能产生符号抵消。实际线性注意力常加非负特征映射、归一化分母、门控或状态缩放；不能只实现 $Q(K^\top V)$ 就认为复现了任一完整论文模型。

**Q14：核化线性注意力的典型归一化形式是什么？**  
A：用特征映射 $\phi$ 表示或近似相似度时，常写成
$$
y_t=\frac{\phi(q_t)^\top\sum_{s\le t}\phi(k_s)v_s^\top}{\phi(q_t)^\top\sum_{s\le t}\phi(k_s)}.
$$
维护分子状态 $S_t$ 与分母状态 $z_t=\sum_{s\le t}\phi(k_s)$ 即可递推；实现仍需处理分母接近零、特征映射的正值性和数值精度。它与 softmax 的关系取决于核/特征设计，不是无条件精确相等。

**Q15：线性注意力的训练与推理“对偶”如何理解？**  
A：同一个因果函数可用块矩阵/并行扫描等形式训练，也可用逐步维护 $S_t$ 的循环形式 decode。课件第 5 页展示“并行的二次形式”与“串行的线性形式”的可切换性；这说明表示等价，不意味着实际训练一定要花 $O(n^2)$——专门的分块/扫描算法可更有效。

**Q16：与标准 KV cache 相比，线性注意力的推理状态改变了什么？**  
A：标准注意力为每层每个历史位置保留 K/V，大小随 $n$ 增长；循环线性注意力保留聚合状态 $S_t$（核化版本还可能有 $z_t$），大小在固定特征维度下与 $n$ 无关。代价是不能像原始 attention 那样任意回看单独历史 token。

**Q17：线性注意力何时不一定比标准注意力快？**  
A：短序列时常数项、额外门控/归一化和专用 kernel 的成熟度可能反转结果；大 $d_kd_v$ 状态也不免费；高效的 FlashAttention 可能使 dense baseline 很强。应分 prefill/decode、长度、batch、精度与硬件实测。

**Q18：RetNet 与线性状态更新是什么关系？**  
A：课件第 5 页只给出一个直观连接：把上一状态乘以衰减因子 $\gamma$，如 $S_t=\gamma S_{t-1}+k_tv_t^\top$，就带有 retention/遗忘行为。实际 RetNet 还有位置相关衰减和具体并行实现，不能把这个单式当作完整定义。

## 3. Mamba-2、Gated DeltaNet 与混合注意力（PDF 第 6–11 页）

**Q19：为什么给循环状态加入输入相关的 $\gamma_t$？**  
A：纯累加状态只能不断叠加历史；$S_t=\gamma_tS_{t-1}+k_tv_t^\top$ 能依输入动态决定保留多少旧信息。$\gamma_t$ 由当前输入产生时，模型具备选择性遗忘能力；但过小会丢失远程信息，过大又容易积累干扰。

**Q20：课件如何概括 Mamba-2 的状态与输出？**  
A：第 7 页写成示意式 $S_t=\gamma_tS_{t-1}+k_tv_t^\top$、$y_t=q_t^\top S_t+v_t^\top D$，其中 $\gamma_t=f(x_t)$，$D$ 是直接通路的系数。这里用线性注意力语言介绍其机制；完整 Mamba-2 是更具体的选择性状态空间模型与结构化状态空间对偶（SSD），实现细节比这两个式子多。

**Q21：式中的 $v_t^\top D$ 起什么作用？**  
A：它提供不必经过递归状态的当前输入直达输出通路，有助于保留局部、即时信息。它不是额外回看全部历史的 attention；维度须由实现中的 $D$ 或逐通道缩放匹配。

**Q22：Mamba-2 为什么仍可能高效训练？**  
A：门控 $\gamma_t$ 可对位置并行计算，带门控的线性递推可转为结构化矩阵或分块扫描运算。串行状态表达适合 decode，并行表达适合训练；关键是实现该变换的 kernel 和数值稳定性。

**Q23：Gated DeltaNet 比简单门控状态多了什么？**  
A：课件第 9 页的式子为
$$
S_t=\gamma_t(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top,\qquad y_t=q_t^\top S_t.
$$
$\gamma_t$ 控制旧状态整体保留，$\beta_t$ 控制当前信息写入与对 $k_t$ 方向的定向修改；后者比仅加 $k_tv_t^\top$ 更像“先改写旧关联，再写入新关联”。

**Q24：如何把 delta 更新式改写成“预测误差写入”？**  
A：先暂取 $\gamma_t=1$，展开得 $S_t=S_{t-1}+\beta_tk_t(v_t^\top-k_t^\top S_{t-1})$。$k_t^\top S_{t-1}$ 是旧状态对当前 key 的预测；$v_t^\top-k_t^\top S_{t-1}$ 是要修正的误差。若 $k_t$ 归一化、$\beta_t=1$，沿 $k_t$ 方向会更接近新 value。一般情形仍受 $\gamma_t$ 和 key 范数影响。

**Q25：为什么说它有“选择性擦除”？**  
A：$(I-\beta_tk_tk_t^\top)S_{t-1}$ 扣除旧状态中由 $k_t$ 方向读出的部分，再加新外积。对于与 $k_t$ 正交的方向，扣除项为零；所以是定向覆盖，不是把整个状态清空。

**Q26：课件说 $\beta_t=0$ 是“no input operation”，是否意味着状态完全不变？**  
A：严格看式子，$\beta_t=0$ 时没有当前 token 的写入或定向擦除，但仍有 $S_t=\gamma_tS_{t-1}$；只有同时 $\gamma_t=1$ 时状态完全不变。面试时应把“无输入写入”与“无状态更新”区分开。

**Q27：线性状态、Mamba-2 和 Gated DeltaNet 的渐进关系是什么？**  
A：线性状态是 $S_t=S_{t-1}+k_tv_t^\top$；加入输入相关 $\gamma_t$ 得到可遗忘的状态；再加入 $\beta_t$ 和 delta 规则，使模型可选择写入并定向改写。更强的状态操作增加表达能力，也带来实现、训练和稳定性的代价。

**Q28：为什么还要保留 full attention 层？**  
A：固定状态很难无损保存任意历史，尤其对精确的长距离检索、复制和多事实绑定可能吃亏。少量 full attention 提供显式访问历史位置的路径，循环层承担多数便宜的局部/统计处理。究竟需要多少 full 层要靠控制变量实验。

**Q29：课件给出哪些混合架构例子？**  
A：第 6 页的 MiniMax-M1 示意约 7 个线性层配 1 个 full attention 层；第 8 页的 Nemotron 3 采用 Mamba/attention 混合；第 10 页的 Qwen 3.5/Qwen Next 示意约 3 个 Gated DeltaNet 层配 1 个 attention 层。这些是各自报告/课件中的方案，不能只凭比例跨模型比较优劣。

**Q30：固定比例 $7:1$ 线性层/full 层，理论上的序列复杂度一定是 $O(n)$ 吗？**  
A：不一定。若那 $1/8$ 的层仍对 $n$ 个位置做完整 dense attention，固定层数下总计算含 $O(n^2)$ 项；“线性增长”可能指某测量区间中循环层或系统因素主导、full 层经过其他约束，或经验曲线近似线性。不能把图上的实测缩放直接当成严格渐近结论。

**Q31：混合比率越偏向线性层，质量一定越差吗？**  
A：不是定律。课件第 11 页提示相关受控消融尚不充分，有些小比例 full 层就能改善 loss/检索；具体还取决于层位置、状态维度、训练数据、任务和 full 层类型。应该比较相同计算预算与训练条件下的质量曲线。

**Q32：full 层放在模型哪里可能重要？**  
A：会影响信息被精确读取后还能经过多少层变换，以及循环层如何在读取前后压缩信息。等间隔、前部、后部或分阶段放置可能不同；课件没有给出普适最佳位置，应作为要实验验证的设计变量。

**Q33：面试中如何解释“状态空间模型与线性注意力的对偶”？**  
A：二者都可将历史对未来的影响写成结构化状态的递推，并可在一定条件下改写为并行矩阵运算。对偶强调**同一算子存在串行与并行视角**，不意味着 Mamba-2 与普通 softmax attention 功能完全相同。

**Q34：线性注意力与 MoE 能组合吗？**  
A：能。它们分别削减序列维度和参数激活维度的成本；如课件中的混合模型会同时使用状态/attention 层与 MoE FFN。不过收益不会简单相乘，因为还受全注意力层、专家通信、内存容量和 kernel 效率约束。

## 4. 动态稀疏注意力 DSA（PDF 第 12–13 页）

**Q35：DSA 与局部滑动窗口注意力最大的区别是什么？**  
A：滑动窗口按**位置**预先限定候选；DeepSeek Sparse Attention（DSA）通过轻量 indexer 根据 query 与历史 token 的内容评分，再选择 top-$k$ 历史位置做昂贵的注意力。动态选择有机会捕捉远距离相关信息，但要支付索引与检索成本。

**Q36：DSA 的两阶段流程是什么？**  
A：先用 lightning indexer 对候选历史位置计算廉价相关分数 $I_{t,s}$，取 top-$k$ 集合 $\mathcal S_t$；再仅对 $s\in\mathcal S_t$ 的 key/value 做主 attention。课件第 12 页截取的论文公式中，indexer 用少量头、低维投影、ReLU 后加权求和产生分数。

**Q37：如果每个 query 仍要给全部历史位置打分，DSA 为何能省钱？**  
A：indexer 虽然仍扫候选位置，但其维度/头数很小，昂贵的高维主 attention 只对 top-$k$ 执行。减少的是**高成本注意力**的配对与访存；总时间是否下降取决于 indexer、top-$k$ 选择、随机 gather 和稀疏 kernel 的开销。

**Q38：能否说 DSA 的总计算严格是 $O(nk)$？**  
A：不能无条件说。若每个位置的 indexer 都扫描全部前缀，索引器本身仍有约 $O(n^2d_{\rm index})$ 工作，而主 attention 约 $O(nk d_{\rm main})$。在 $d_{\rm index}\ll d_{\rm main}$ 的相关长度区间可有显著实测收益，但“主 attention 稀疏”不等于“整个系统严格线性”。

**Q39：DSA 为什么不一定直接把 KV cache 变成 $O(k)$？**  
A：每个新 query 的 top-$k$ 历史位置可能不同；为能选到任意过去 token，通常仍需保存或能访问历史候选表示。主 attention 一次只读取 $k$ 个位置，与所有历史候选的存储/索引是不同问题。若再做压缩、淘汰或分层存储，才可能改变缓存规模。

**Q40：动态 top-$k$ 会引出什么训练问题？**  
A：选择索引本身离散，容易出现梯度稀疏或选错目标、训练初期不稳定；indexer 需学会近似主注意力的相关性，可能有 warm-up/蒸馏阶段。具体训练法以 DeepSeek-V3.2 原报告为准；课件只强调可在 dense 短上下文预训练后做后续适配。

**Q41：课件第 13 页的性能图应怎么读？**  
A：图展示 DeepSeek-V3.2 与其他模型的若干基准，以及长上下文 prefill/decode 价格或吞吐比较，属于不同模型/服务条件的**报告结果**，不能单独证明 DSA 是全部差异的原因。与第 12 页的架构机制分开记，更适合回答“如何设计公平消融”。

**Q42：面试官问“DSA 与线性注意力哪个更适合精确检索”，怎么答？**  
A：DSA 保留了对选中历史 token 的显式 softmax/主 attention，有利于精确读取，但可能漏掉未被 indexer 选中的关键位置；固定状态线性注意力不显式保存每个位置，可能因压缩产生干扰。应在相同资源预算、长上下文任务上评估，而不是仅靠复杂度选。

## 5. MoE 的基本结构与经济账（PDF 第 14–25 页）

### 先弄清“参数多”和“算得多”不是同一件事

常见 Transformer MoE **替换 FFN/MLP 子层**，而非把整层的 attention、embedding 和输出头都复制 $N$ 份。设专家 $E_i(x)$，router 给分数 $s_i(x)$，选择集合 $\mathcal T(x)=\operatorname{TopK}(s(x),k)$，典型输出可抽象成：

$$
\operatorname{MoE}(x)=\sum_{i\in\mathcal T(x)}g_i(x)E_i(x),
$$

若有 shared experts，再加 $\sum_{j=1}^{N_s}E_j^{\rm shared}(x)$；实际模块外还保留残差、归一化等。$g_i$ 是门控权重，是否归一化、在 top-$k$ 前还是后做 softmax/sigmoid，由模型决定。以每个专家约 $P_e$ 参数估算，仅专家部分：

$$
P_{\rm stored}\approx (N+N_s)P_e,\qquad
P_{\rm used/token}\approx(k+N_s)P_e.
$$

这说明**固定 $k$ 增加 $N$，理想的每 token 专家主计算不随 $N$ 同比例增加**，但显存、参数同步、router、跨设备通信和负载均衡仍可能增加。对整个模型还要加共享 attention、embedding 等参数与计算。

**Q43：MoE 的“专家”通常是什么？**  
A：在本讲主线中，多数专家是相互独立的 FFN/MLP（通常与 Transformer 的 FFN 形状相近），token 通过 router 只送给少数专家。也存在给 attention head 做专家选择的研究，但课件第 25 页指出这不是最常见形式。

**Q44：MoE 中“稀疏”具体稀疏在哪里？**  
A：是**计算路径稀疏**：全部专家权重存在，但一个 token 只计算其中 $k\ll N$ 个 routed experts。不是参数矩阵本身每个元素大多为零，也不等于 attention 沿 token 维度稀疏。

**Q45：为什么总参数大很多，FLOPs 却不必同比增加？**  
A：因为每个 token 只激活固定数目的专家。理想情况下，把 $N$ 从 8 增至 64 而 $k=2$ 不变，每 token 只跑 2 个 routed FFN；增加的是可选容量。此说法忽略 router 打分、专家可能变小、跨卡通信等附加成本，且训练优化器仍要存全部参数相关状态。

**Q46：“同 FLOPs 更多参数可能更好”有什么前提？**  
A：要让专家真正学到互补功能，数据足以训练每个专家，路由没有塌缩，且硬件能高效放置/读取所有权重。第 16 页 Switch 图展示特定设置下专家增多带来更低 loss；不能推断专家越多永远越好。

**Q47：为什么第 17 页说 MoE 可能更快训练？**  
A：对某个目标 loss/指标，稀疏模型可能用更少的**训练时间或 token/FLOPs**到达；课件列出 Switch、OLMoE 的示例结果。但 per-step 比较可能相反：MoE 的通信和小矩阵计算可能使一步更慢。因此“更快达到目标”和“单步更快”要分别测。

**Q48：$P_{\rm total}$、$P_{\rm active}$ 与显存的关系如何？**  
A：$P_{\rm active}$ 更贴近每 token 计算量；$P_{\rm total}$ 决定权重必须在某处存放，训练时通常还要梯度/优化器状态。分卡部署可降低单卡持有量，但增加专家分发通信。故“37B active”不等于“单卡只需存 37B 权重”。

**Q49：MoE 推理 latency 为什么可能比同激活参数的 dense 模型差？**  
A：expert dispatch/gather、跨设备 all-to-all、路由不均衡与很多小 GEMM 的低利用率可能超过省下的主乘法；decode batch 小时尤其敏感。比较需区分单请求延迟、吞吐、batch 大小、总权重是否常驻显存和专家并行拓扑。

**Q50：课件为什么强调 MoE 适合多设备？**  
A：不同专家可放到不同加速器上，让每台设备只持有部分专家；token 经路由分发到对应设备，专家并行提供额外容量维度。但需要通信互联与足够负载，否则理论上的并行不能变成 wall-clock 收益。

**Q51：为什么 MoE 没有从一开始就全面取代 dense 模型？**  
A：第 24 页列举基础设施复杂、收益较依赖多设备/高吞吐环境，以及训练目标含启发式平衡项、可能不稳定。现实还要处理 capacity、丢 token、通信、推理小 batch 和微调过拟合。MoE 是资源与质量的一个 Pareto 选择，不是无代价升级。

**Q52：第 18、20–23 页的跨模型性能图能证明“MoE 必然优于 dense”吗？**  
A：不能。模型常同时改变训练数据、token 数、蒸馏/后训练、架构、规模和评测方式。图能说明 MoE 已具强竞争力，但要归因某个 MoE 设计，需要同数据同预算的受控消融。课件第 33–34 页更接近此类证据。

**Q53：MoE 何时最值得考虑？**  
A：训练/服务环境能承载大量总权重，有较高吞吐、良好跨设备网络，且希望以有限每 token 计算扩展容量时。若单机显存紧、低并发、小模型或极低 latency 是硬约束，复杂的 MoE 未必合适。

**Q54：一个 $N=64,k=8,N_s=1$ 的模型激活了多少专家？**  
A：每 token 激活 8 个 routed experts 和 1 个 shared expert，合计 9 个专家计算路径；但不能说“激活参数占比就是 $9/65$”，除非所有专家同大小且忽略模型其他共享组件。

**Q55：MoE 的专家是否一定具有清晰的人类可读分工？**  
A：不一定。训练可能出现统计上的专门化，但 route 可以按 token 频率、语境或其他分布划分，也可能出现冗余、热门专家或塌缩。需要用路由统计、消融和任务分析验证，不能把“专家”一词理解成预定义知识领域。

**Q56：MoE 能否减少 attention 的长上下文二次成本？**  
A：单独的 FFN-MoE 不改变 attention 的 $QK^\top$ 配对；它主要改变 FFN 容量/激活计算。若要解决长上下文，还需与线性、稀疏、窗口或缓存压缩等注意力方案组合。

## 6. 路由、门控、细粒度与共享专家（PDF 第 26–35 页）

**Q57：课件把 MoE 设计变量分成哪三类？**  
A：路由函数（谁选谁、top-$k$ 怎么选）、专家尺寸/数量（细粒度和 shared experts）、训练目标（平衡、稳定等）。三个变量相互影响，例如专家越细，路由与通信开销相对越显著。

**Q58：token-choice、expert-choice、全局匹配有什么区别？**  
A：token-choice 是每 token 选若干专家，容易保证每 token 有 $k$ 个计算路径，但专家负载可能偏斜；expert-choice 是每专家选若干 token，容易控制容量，但每 token 被选次数可能不同；全局匹配联合分配 token/专家，可强制容量约束，通常更复杂。

**Q59：为什么大多数模型仍使用 token-choice top-$k$？**  
A：计算和实现较直观、每 token 的激活数明确、可用成熟的 dispatch/合并流程。代价是需要平衡机制和容量管理。课件第 28 页的实验给了 token-choice 与 expert-choice 的某些差异，但不能把某一次实验外推成定律。

**Q60：top-$k$ 路由的最基本算法是什么？**  
A：对 token 表示 $x_t$ 计算各专家分数 $z_{i,t}=x_t^\top e_i$（或其他打分函数）；取最大 $k$ 个专家索引 $\mathcal T_t$；仅计算这些专家的 $E_i(x_t)$；以门控权重 $g_{i,t}$ 加权求和并接回残差。第 31 页给出了 DeepSeek V1/V2 风格的公式。

**Q61：router 的 softmax 和 top-$k$ 谁先谁后，为什么要区分？**  
A：先对全部 $N$ 个分数 softmax，再保留 top-$k$，所选权重一般**和小于 1**；先 top-$k$ 再仅在选中者上 softmax，权重和为 1。二者输出尺度与梯度不同；第 31 页指出 DeepSeek V1/V2、Grok、Qwen 与 Mixtral/DBRX/DeepSeek V3 的一些路由归一化顺序不同，具体模型还须看原报告。

**Q62：给一个“先 softmax”和“后 softmax”的数字例子。**  
A：logits 为 $(2,1,0)$、$k=2$。全 3 专家 softmax 约为 $(0.665,0.245,0.090)$，保留前两个后权重和约 $0.910$；若只在前两个上 softmax，变成约 $(0.731,0.269)$，和为 1。即使 top-$k$ 索引相同，输出尺度也不同。

**Q63：top-$k$ 决策不可导，为什么 router 还能训练？**  
A：离散**索引选择**在切换边界不可导，但被选中的门控权重 $g_i$ 对 router 分数通常仍可导，损失可经这些连续权重反传。未选专家的梯度情况依赖归一化方式；比如先全局 softmax 时，分母也可能让未选 logits 受影响。只是这种梯度不等于对离散选择的精确梯度。

**Q64：top-1 与 top-2 的主要取舍是什么？**  
A：top-1 减少专家主计算和跨卡路径，吞吐潜力较高；top-2 提供两个专家的组合，更可能缓解单一路由失误，但增加计算与通信。质量、稳定性与成本必须一起看。

**Q65：哈希路由为何是有用的 baseline？**  
A：它绕过学习式 router，以固定规则分配 token，可帮助判断收益究竟来自更多专家容量，还是来自学到的动态选择。通常不是最灵活方案，但更容易分析分配与复现。

**Q66：RL 或全局匹配路由为什么没有成为默认选项？**  
A：RL 可以直接优化离散决策，却有高方差、额外训练复杂度；全局匹配可强制均衡，但求解/通信开销高。课件第 30、37 页介绍它们作为备选研究方向，实际大量系统仍选简单 top-$k$ 加辅助机制。

**Q67：什么是“细粒度专家”？**  
A：将原来较大的 routed FFN 切成更多、每个更小的专家，并相应增加每 token 选中的专家数，尽量控制总激活计算。这样可形成更灵活的组合，但路由、分发和小矩阵效率可能更难。

**Q68：为什么将大专家拆小后仍能保持相近 FLOPs？**  
A：假设原来选 $k$ 个、每个专家大小 $P_e$；拆成 $m$ 倍数量、单个约 $P_e/m$，每 token 选约 $mk$ 个，则激活专家参数仍约 $mk(P_e/m)=kP_e$。这是粗略预算，实际层宽取整、门控、通信和 kernel 效率不会严格相同。

**Q69：shared expert 与 routed expert 的职责区别是什么？**  
A：shared expert 每个 token 都执行，倾向承接所有 token 都用得上的通用变换；routed expert 只服务被选择的 token，承接条件化容量。这是设计意图，不代表训练后一定出现完全无重叠的语义分工。

**Q70：为什么不能笼统说“shared experts 一定提高质量”？**  
A：课件第 33 页 DeepSeekMoE 消融中，共享专家与细粒度有收益；第 34 页 OLMoE 消融则显示细粒度有收益，而 shared expert 未带来明显额外增益。结果依赖基线、参数/FLOPs 匹配、训练数据和路由细节。

**Q71：阅读课件第 35 页表格时，最容易误读什么？**  
A：`Routed` 是可选择的 routed 专家总数，`Active` 通常是每 token 激活的 routed 专家数，`Shared` 是无条件执行的共享专家数。不能将表中的 `Active` 当作所有激活专家总数，也不能仅用专家个数推断总/激活参数。

**Q72：为什么各模型 top-$k$ 的表格数字不能直接当成性能排序？**  
A：它们的单个专家大小、shared 专家、attention、层数、总训练 token 和系统部署均不同。`k=8` 的细粒度方案可能与 `k=2` 的大专家方案激活计算相近；先归一化参数与计算再比较。

**Q73：什么是 routing collapse？**  
A：router 长期将大量 token 送往少数专家，使部分专家过载、其他专家几乎没训练，浪费容量并可能触发丢 token/延迟尖峰。可通过专家使用频率、负载方差、每专家 token 数、丢弃率和 loss 监控。

**Q74：router 的分数一定要经 softmax 吗？**  
A：不一定。DeepSeek V3 使用 sigmoid 得到逐专家 affinity，在被选中专家之间再归一化；其他模型使用 softmax logits。sigmoid 分数不要求所有专家在选择前形成一个总和为 1 的分布。

**Q75：为何不能仅用 “top-$k$ 不可导” 解释 MoE 的全部训练困难？**  
A：即使 router 可从选中权重收到梯度，还会有负载不均、专家训练数据不足、容量溢出、通信瓶颈和低精度数值稳定性。课件后半重点实际上是这些优化与系统约束。

**Q76：面试官让你设计路由消融实验，应怎么做？**  
A：固定总参数、激活参数/每 token FLOPs、训练数据与 token 数、专家放置和硬件，逐一比较 top-1/top-2、softmax 顺序、细粒度、shared expert、平衡强度；记录质量、路由熵/负载、丢 token 和 wall-clock。否则可能把“更大容量”错归因于“更好路由”。

## 7. MoE 路由训练、负载均衡与 z-loss（PDF 第 36–43、48–49 页）

### 必须会推的 Switch 负载均衡式

第 40 页给出 Switch Transformer 的 top-1 辅助目标。对 batch 的 $T$ 个 token，$p_i(x)$ 是专家 $i$ 的 router 概率，$f_i$ 是**实际被分发**到专家 $i$ 的 token 比例，$P_i$ 是**平均预测概率**：

$$
f_i=\frac1T\sum_{x\in B}\mathbf 1[\arg\max_j p_j(x)=i],\qquad
P_i=\frac1T\sum_{x\in B}p_i(x),\qquad
L_{\rm bal}=\alpha N\sum_{i=1}^{N}f_iP_i.
$$

若两者都均匀，即 $f_i=P_i=1/N$，则 $L_{\rm bal}=\alpha$。因 $f_i$ 由硬选择计算，通常把它视为当前 batch 的非可导统计量；在这一约定下，$\partial L_{\rm bal}/\partial p_i(x)=\alpha Nf_i/T$。热门专家的 $f_i$ 大，最小化时其概率受到较强下压。这个辅助项鼓励均衡，但并不保证每个 batch 都均匀，也不是唯一可行机制。

**Q77：top-$k$ 路由不连续，常见训练办法有哪几类？**  
A：课件列三类：用 RL 直接优化离散选择、加入随机扰动帮助探索、以及使用平衡等启发式辅助目标。实际系统常以可微门控权重配合平衡机制为主，而不是为每次路由完整运行策略梯度。

**Q78：REINFORCE 解决了哪个理论问题，又引入什么实际问题？**  
A：它可用采样路由的奖励/损失估计离散选择策略的梯度，不要求 top-$k$ 索引可导。但估计方差较高，常需 baseline、更多样本与复杂训练流程；第 37 页显示其可工作，却不显著到成为明确默认方案。

**Q79：Shazeer 等的 noisy top-$k$ 路由如何做？**  
A：在 token-专家的原始打分上加可学习尺度的高斯噪声，再取 top-$k$、对保留者归一化。第 38 页示意为 $H(x)_i=(xW_g)_i+\epsilon_i\operatorname{softplus}((xW_{\rm noise})_i)$，$\epsilon_i\sim\mathcal N(0,1)$。噪声鼓励探索和一定鲁棒性，但仍要权衡训练波动。

**Q80：Switch 中的 input jitter 与高斯 noisy top-$k$ 是一回事吗？**  
A：不是。第 39 页显示给 router 输入或 logits 加轻微均匀扰动，属于更简单的随机化；高斯 noisy top-$k$ 还学噪声尺度。课件指出 ST-MoE 后续实验移除了某种 jitter，说明其收益依设置而定。

**Q81：为什么系统效率要求专家尽量均衡？**  
A：专家并行时，单步要等最慢/最拥挤的设备或专家完成；热门专家会出现队列和 capacity 溢出，冷门专家计算资源闲置。均衡不仅是“公平”，直接关系吞吐、尾延迟和专家能否充分训练。

**Q82：$f_i$ 与 $P_i$ 为什么要分开定义？**  
A：$f_i$ 是离散的**实际分配频率**，$P_i$ 是可微的**平均偏好**。用 $f_iP_i$ 可以让实际忙的专家的预测概率下调；若只比较 $P_i$，可能无法充分反映硬 top-1 后的真实负载。

**Q83：为何 Switch 辅助损失在均匀分配时等于 $\alpha$，而非 0？**  
A：把 $f_i=P_i=1/N$ 代入：$\alpha N\cdot N(1/N^2)=\alpha$。这个目标看重相对变化和梯度方向，不要求理想点数值为 0。不能因为最小值非零就误认为公式错误。

**Q84：从导数怎样看出“热门专家被压制”？**  
A：在视硬路由统计 $f_i$ 为常量时，$\partial L_{\rm bal}/\partial p_i(x)=\alpha Nf_i/T$。若专家 A 的 $f_A$ 大于 B，同样增加其概率带来更高惩罚；梯度下降倾向抑制 A 的概率、推动分配分散。真实参数梯度还需经 softmax 链式法则，不可直接把这个偏导当作 logits 的偏导。

**Q85：top-$k$ 版本的平衡统计如何改写？**  
A：每 token 有 $k$ 次 routed 分配，可用 $c_i=\sum_t\mathbf1[i\in\mathcal T_t]$ 计数；定义 $f_i=Nc_i/(kT)$，使均匀时 $f_i\approx1$，再配平均 affinity/probability $P_i$。不同论文的常数、归一化和 shared 专家处理不同，不能把 Switch top-1 公式未经调整直接照搬。

**Q86：DeepSeek V1/V2 的 per-expert 平衡与 Switch 有何相似处？**  
A：第 41 页给出的 $L_{\rm ExpBal}=\alpha_1\sum_i f_iP_i$ 使用实际选中频率与平均 router score 的乘积，思路相同；因是 top-$k$，$f_i$ 的规范化带 $N_r/(kT)$。具体 $P_i$ 是否来自 softmax/affinity 应按该模型公式读。

**Q87：per-device 平衡的目的是什么？**  
A：把同一设备上的多个专家负载聚合，避免“单专家大致平均但某台设备整体过载”，从而减轻设备间完成时间不一致和通信热点。它与 per-expert 平衡关注的粒度不同，常需同时看。

**Q88：为什么平衡辅助损失不能设得越大越好？**  
A：过强的辅助项可能迫使 token 选不适合的专家，损害主语言模型目标；过弱则会负载失衡。需共同观察主验证 loss、专家分配、丢 token、吞吐，并通过消融选权重。

**Q89：DeepSeek V3 的 per-expert bias 如何影响路由？**  
A：给专家 $i$ 一个偏置 $b_i$，用 $s_i+b_i$ 决定是否进入 top-$k$；但被选后的输出门控权重仍由原始 $s_i$ 计算，不把偏置直接乘进专家输出。过载专家的 $b_i$ 下调、欠载专家上调，以在线反馈控制负载。

**Q90：DeepSeek V3 的 bias 是普通可训练参数吗？**  
A：课件与原报告介绍的是基于每步负载统计的**在线更新**：若过载，偏置减一个步长；若欠载，加一个步长。它用来调度选择，不应笼统等同于由语言模型 loss 的普通反向传播学习的权重。

**Q91：“auxiliary-loss-free load balancing” 是否真的没有任何平衡辅助 loss？**  
A：不是字面上的完全没有。DeepSeek V3 主要用在线 bias 做**跨 batch 的专家负载控制**，同时保留很小的 **sequence-wise** 辅助平衡损失，防止单条序列内部极端不均衡。课件第 42 页明确提醒“not fully aux loss free”。

**Q92：sequence-wise 辅助损失解决什么边界情况？**  
A：即使整个 batch 的专家计数均衡，一条序列内仍可能集中路由到少数专家。sequence-wise 项按序列计算分配频率与 router 偏好，缓解局部极端偏斜；不能用全局均衡统计替代检查。

**Q93：为什么不用输出门控 $s_i+b_i$，而只用它选专家？**  
A：若把负载控制偏置也乘入输出权重，在线调整负载时会直接扰动专家贡献幅度与主任务函数。把“选谁”与“选中后贡献多少”分离，可以用偏置调度负载，同时让贡献由原始内容 affinity 决定。

**Q94：课件第 43 页“去掉负载均衡 loss”的图说明什么？**  
A：OLMoE 设置中，无 LBL 时专家分配明显更偏斜，训练/验证曲线也有所差异；加 LBL 后负载更均匀。这是该模型的消融，不与 DeepSeek V3 的在线 bias 结果冲突——两者使用的平衡机制并不同。

**Q95：若监控发现专家 0 收到 50% token，其他专家很少，优先检查什么？**  
A：先查 router logits/softmax 数值、top-$k$ 选择和 capacity/drop 统计；再查平衡 loss/bias 是否正确生效、专家设备映射、数据是否高度偏斜。仅调大平衡系数可能掩盖代码错误或牺牲质量。

**Q96：router 为什么比普通 FFN 更易有低精度数值问题？**  
A：router 的微小分数差会改变离散 top-$k$ 索引；softmax 的大 logits 还可能因低精度舍入导致概率突变。第 48 页引用 ST-MoE 的例子说明，用 bf16 表示接近的较大数时，舍入可改变原本不同的分数排序或比例。

**Q97：为什么有些系统让 router 用 float32，其余仍用低精度？**  
A：router 参数/运算占主模型一小部分，提升其数值精度通常成本较低，却能稳定 softmax、top-$k$ 和梯度；不必把所有专家 GEMM 一起改为 fp32。具体方案还需看混合精度实现。

**Q98：router z-loss 的形式及目标是什么？**  
A：第 48 页给出 $L_z=\frac1T\sum_t(\log\sum_i e^{z_{t,i}})^2$，训练时乘正系数加到主损失。它约束 router 的 log-partition/logit 整体尺度，降低极端大 logits 导致的数值问题；它不是标准交叉熵，也不是 top-$k$ 平衡损失。

**Q99：z-loss 与“softmax 减最大值”有何区别？**  
A：减最大值是计算 softmax 的等价数值技巧，不改变数学概率；z-loss 是**新增训练目标**，会改变参数梯度。即使实现了稳定 softmax，极端 logits 仍可能影响路由训练，因此两者可并用。

**Q100：第 49 页去掉 z-loss 的图应怎样总结？**  
A：在所示 OLMoE 消融中，无 router z-loss 出现更明显的 loss/指标尖峰；加入后更稳定。它提示 router 精度与 logit 规模的重要性，但 z-loss 的最优系数不能直接迁移到所有模型。

**Q101：负载均衡 loss 与 router z-loss 各管什么？**  
A：前者主要约束**专家使用分布**，降低热专家/冷专家失衡；后者主要约束**router logits 的数值尺度**。一个模型可能路由均衡但 logits 过大，也可能 logits 稳定但集中使用少数专家，需要分别诊断。

**Q102：面试官问“top-$k$ 不可导，那平衡 loss 怎么对 router 有梯度”，如何答？**  
A：分配频率 $f_i$ 可以当作当前 batch 的非可导常量，但平均概率 $P_i$ 或 affinity 是可微函数；$L_{\rm bal}$ 经 $P_i$ 将偏好反馈给 router。它不是在硬选择节点直接反传精确梯度。

**Q103：怎么验证负载均衡机制没有暗中伤害模型质量？**  
A：记录主 loss、验证 loss、任务指标、每专家使用直方图、drop rate、训练吞吐和延迟；做无平衡、不同系数、在线 bias 的消融，保持数据/FLOPs/总参数与随机种子尽量可比。只看专家直方图“好看”不够。

## 8. 专家并行、通信、token dropping 与微调（PDF 第 44–50 页）

**Q104：专家并行（EP）的基本数据流是什么？**  
A：本地 token 经 router 产生专家索引；按目标专家/设备重排并 all-to-all dispatch；每个设备对收到的 token 执行本地专家 FFN；输出再 all-to-all 返回原 token 所在设备，按门控权重合并并恢复顺序。路由计算少并不代表传输少。

**Q105：EP 与数据并行（DP）有什么区别？**  
A：DP 复制模型、把不同数据分到设备，通常同步梯度；EP 把不同专家权重放在不同设备上，token 要流向相应专家。大型训练常组合 DP、EP、张量并行和流水线并行，通信模式各不相同。

**Q106：为何 MoE 推理/训练都可能被 all-to-all 限制？**  
A：token 按内容动态选择不同设备上的专家，dispatch 与结果回传产生跨卡/跨节点通信；网络带宽、拓扑、消息粒度和负载偏斜都影响时间。FFN 理论乘法减少，若通信成为主瓶颈，wall-clock 仍可能不理想。

**Q107：什么是 expert capacity？**  
A：为每个专家在一个 batch/路由组中预留的最大 token 数。平均期望负载约为 $kT/N$ 次分配，实际容量可设为 $C\approx\lceil c\,kT/N\rceil$，$c$ 为 capacity factor。负载波动超过 $C$ 会触发丢弃、重路由或动态扩容；容量设太大又造成 padding 浪费。

**Q108：token dropping 与 padding 的取舍是什么？**  
A：固定容量小便于高效批量矩阵乘，但热门专家超额 token 可能不经专家路径；容量大能少丢 token，却给冷门专家填大量空位、浪费运算与显存。块稀疏计算试图同时减少丢弃和无效 padding。

**Q109：第 45 页 MegaBlocks 的关键思想是什么？**  
A：把不同专家接收到的可变长度 token 批次表示为块稀疏矩阵乘，以适配动态负载；相较于固定容量 padding 或超额 drop，能更灵活处理不均衡。图中从 batched GEMM、块对角 GEMM 到 block-sparse GEMM 的变化，重点是**系统布局**，并非改写 MoE 数学目标。

**Q110：为什么小专家不一定更快？**  
A：细粒度增加可组合性，但每个专家的 token 批次可能很小，小 GEMM 难以吃满 GPU；分发次数、元数据和 kernel 启动开销上升。应按硬件适配专家宽度、批量大小和块稀疏实现。

**Q111：如何用通信量粗估 EP 的代价？**  
A：一个 batch 有 $T$ 个 token、每 token 选 $k$ 个远端专家、隐藏维度 $d$、元素 $s$ 字节，则仅 dispatch 的激活传输量可粗估为 $O(Tkds)$，结果返回再同量级；实际取决于本地命中率、复制/合并方式、网络拓扑与通信压缩。这个估计不是训练全部通信量。

**Q112：课件第 46 页“latent MoE”如何降低通信？**  
A：在跨设备分发前把隐藏激活下投影到较低维，在专家计算/汇总后再上投影，降低 all-to-all 字节数；代价是额外投影计算和信息瓶颈。它与后文 MLA 的“压缩 KV latent”都用了低维表示，但作用位置不同，不能混为一谈。

**Q113：为什么 MoE 可能出现与 batch 其他请求有关的随机性？**  
A：若专家 capacity 在整个 batch 上共享且溢出时 drop token，某个请求是否超额取决于同批其他请求也被路由到哪些专家。即使自己的输入完全相同，改变混批可能改变其专家路径与输出。不是所有现代 MoE 实现都会 drop，但需要检查部署策略。

**Q114：这个 batch 依赖对线上服务意味着什么？**  
A：可能影响复现性、缓存命中和输出稳定性；线上需明确是否用 capacity/drop、是否跨用户混批、溢出时如何处理，并用固定模型/相同输入在不同 batch 组合下测试。不应仅凭“MoE”标签就断言一定发生此现象。

**Q115：推理时是否还需要负载均衡？**  
A：需要考虑，但目标与训练不同。训练强调每个专家得到足够梯度且并行高效；服务强调尾延迟、吞吐、热门专家复制/放置和网络热点。训练时的平衡 loss 不能替代线上负载管理。

**Q116：为何 sparse MoE 在小数据微调时可能过拟合？**  
A：模型总容量高、不同专家被训练到的 token 数有限；小微调集可能只强烈更新部分专家和 router，导致专门化过头或路由漂移。第 50 页是相关实验证据，不是所有任务必然如此。

**Q117：课件给出哪些微调应对思路？**  
A：ST-MoE 相关结果展示仅微调非 MoE 的 MLP/部分 dense 参数可能更稳；DeepSeek 案例使用大量 SFT 样本。实际还可把更新范围、学习率、正则化和路由冻结与否做消融，但这些是延伸建议，不是课件证明的统一解法。

**Q118：如果 GPU 利用率低但理论 FLOPs 很少，应该先排查什么？**  
A：查看 router/重排/all-to-all/GEMM/合并的时间分解、每专家实际 batch 大小、负载方差、padding/drop、跨节点通信与 kernel 数量。理论激活 FLOPs 少，可能只是说明性能瓶颈转到了数据搬运与小矩阵。

## 9. Upcycling：从 dense checkpoint 出发（PDF 第 51–53 页）

**Q119：什么是 sparse upcycling？**  
A：从已有 dense 模型 checkpoint 初始化 MoE：保留可复用的 embedding、attention、norm 等参数；把 dense FFN 复制或映射到多个专家；初始化新 router，再继续训练。它复用已经投入的 dense 预训练成本，而不是从零训练全部专家。

**Q120：如果把同一个 FFN 复制到所有专家，最初的 MoE 输出是否与原 FFN 相同？**  
A：在**没有额外 shared 输出**且选中专家门控权重和为 1 的条件下，若所有专家初始权重完全相同，则 $\sum_i g_iE_i(x)=E(x)$，可保持原函数。若权重和不为 1、增添 shared expert 或改动残差尺度，就不严格等价，需重新缩放/校准。

**Q121：复制出来的专家怎样学会不同功能？**  
A：不同 token 通过 router 进入不同专家，各专家逐渐接受不同梯度；训练噪声、门控与数据分布使其分化。若路由完全塌缩或复制后对称性从未打破，容量不会被有效利用；所以初始化、平衡和继续预训练仍重要。

**Q122：upcycling 与单纯增加模型参数有什么区别？**  
A：它从已经训练好的 dense 函数出发，增加稀疏可选容量，希望在继续训练时保留基础能力并获得新收益；不是把新增专家随机初始化后宣称“预训练完成”。比较时需计入原 dense 预训练投入与 upcycling 后的追加 token/时间。

**Q123：课件里的 MiniCPM-MoE 与 Qwen MoE 例子说明什么？**  
A：第 52 页的 MiniCPM-MoE 使用 top-2、8 个专家，继续训练后相对 base 展示若干指标收益；第 53 页的 Qwen1.5-MoE 从 Qwen 1.8B 初始化，60 routed、4 shared、top-4，并报告与若干 dense/MoE 模型比较。它们证明 upcycling 有可行案例，不能把各指标提升直接归因于“复制 FFN”单一操作。

**Q124：upcycling 时应该记录哪些对照组？**  
A：原 dense checkpoint；相同额外 token/FLOPs 的 dense 继续训练；同总/激活参数的从零 MoE；不同 router/平衡初始化的 upcycled MoE。这样才能拆开“继续训练更多数据”和“MoE 架构”两个效应。

## 10. DeepSeekMoE V1→V3、MLA 与 MTP（PDF 第 54–59 页）

| 课件页 | 模型 | 课件给出的总量/激活量 | 主要路线 | 阅读提醒 |
| --- | --- | --- | --- | --- |
| 54 | V1 | 约 16B / 2.8B | 细粒度 routed + 共享专家；top-$k$ 与专家/设备平衡 | 数值按课件与原报告语境理解。 |
| 55 | V2 | 约 236B / 21B | 160 routed、其中 6 active，另有 2 shared；限制目标设备、通信平衡；MLA | `active` 不是仅 FFN 计算的参数。 |
| 56 | V3 | 约 671B / 37B | 256 routed、其中 8 active，另有 1 shared；sigmoid/选中后归一化、在线 bias 和轻量序列平衡；MLA/MTP | 该页有两处数字/标签笔误，见第 12 节。 |

**Q125：DeepSeekMoE V1 的两项核心专家设计是什么？**  
A：把专家细分以提供更多组合可能，并把少数专家隔离为每 token 都调用的 shared experts。课件第 54 页在此基础上展示常规 top-$k$ 与专家/设备辅助平衡。

**Q126：DeepSeek V2 相对 V1，MoE 路由的系统重点是什么？**  
A：第 55 页强调 top-$M$ 设备/组限制和 communication balancing：先限制 token 可去的专家所在设备，再从这些候选中选专家，以减少跨节点扇出；同时控制不同设备发送/接收的通信量。它是在质量与网络成本之间增加约束。

**Q127：为什么要同时关注“发送出去”和“接收到”的通信？**  
A：某设备既可能把本地 token 发往多个专家设备，也可能作为专家宿主收到其他设备的 token。只均衡专家计算而忽略跨设备消息方向，仍会出现网络拥堵与尾部拖慢。

**Q128：DeepSeek V3 的 routed 专家分数与最终权重如何计算？**  
A：原报告中先以 $s_{i,t}=\operatorname{sigmoid}(u_t^\top e_i)$ 得到 affinity；按可能加 bias 的分数挑出 8 个 routed experts；然后对**选中者的原始** $s_{i,t}$ 做归一化并加权其 FFN 输出。shared expert 另行始终执行。

**Q129：第 35、56 页专家数量读起来不一致，应该记哪个？**  
A：第 35 页写 V3 有 **256 routed + 1 shared**；第 56 页写“258 experts”且标题下误写“V2”。DeepSeek-V3 原论文的配置明确是 **256 routed、每 token 激活 8、1 shared**；复习时以原论文为准，把课件第 56 页视为笔误。

**Q130：DeepSeek V3 的 671B/37B 是否等于“每 token 只算 37/671 的时间”？**  
A：不能这么算。总参数量/激活参数量不是 wall-clock 比例；注意力、路由、专家通信、缓存、硬件利用率以及不同参数的重复使用都影响延迟。37B 也包含共享组件，不是 8 个专家权重之和。

**Q131：MLA（Multi-head Latent Attention）要解决什么问题？**  
A：标准 MHA 的 KV cache 随历史 token 数和 KV 头/维度增长；MLA 将 K/V 的主要内容先投影成较低维 latent $c_t^{KV}$，推理时主要缓存该 latent，从而减小缓存/访存。它不是 MoE 路由，处理的是 attention 的 K/V 表示。

**Q132：MLA 的基本低秩分解如何写？**  
A：示意为 $c_t^{KV}=W^{DKV}h_t$，$k_t^C=W^{UK}c_t^{KV}$，$v_t^C=W^{UV}c_t^{KV}$。$W^{DKV}$ 下投影，$W^{UK},W^{UV}$ 上投影。若低秩维度 $d_c$ 显著小于直接存储的全部 K/V 维度，缓存 latent 可省空间；但需计入额外 RoPE key 分支等实际缓存。

**Q133：为什么不一定要先显式还原完整的 $k_t^C$ 再做 QK 点积？**  
A：没有位置旋转时，$q_t^\top k_s^C=q_t^\top W^{UK}c_s^{KV}=((W^{UK})^\top q_t)^\top c_s^{KV}$，可把上投影矩阵吸收到 query 一侧。这样历史只需提供 latent，减少 decode 时对大 K 向量的存取。

**Q134：RoPE 为什么与这种“吸收 K 上投影”有冲突？**  
A：query/key 会受各自位置相关旋转 $R_t,R_s$ 作用，分数含 $q_t^\top R_t^\top R_sW^{UK}c_s^{KV}$；其中 $R_s$ 随历史位置 $s$ 变化，无法把一个固定的 $W^{UK}$ 简单吸收进当前 query 并完全消掉历史位置相关部分。这里的难点是位置依赖，不是矩阵乘法结合律本身失效。

**Q135：DeepSeek MLA 如何兼顾 latent 缓存与 RoPE？**  
A：将 key/query 的内容部分与位置旋转部分解耦：内容部分利用压缩 latent 和矩阵吸收，另用少量额外的 RoPE key/query 维度表达位置；推理通常缓存 $c_t^{KV}$ 及小的旋转 key，而不是字面上“只缓存一个 latent、绝无其他缓存”。

**Q136：MLA 与 GQA 的节省方式有什么不同？**  
A：GQA 让多个 query 头共享较少的完整 KV 头；MLA 用低秩 latent 压缩 K/V 内容，并通过数学重排减少重建/缓存需求。两者都可降低 KV 压力，但参数化、注意力打分形式、RoPE 处理和 kernel 要求不同。

**Q137：MTP（multi-token prediction）在此课件中指什么？**  
A：主模型按常规预测下一个 token，附加轻量预测模块在训练时预测更远的未来 token，以提供更丰富的监督。课件第 59 页说 DeepSeek V3 的额外 MTP 深度为 1，即除了通常的下一 token 之外，**再预测多一个未来 token**；不是一次训练就不再自回归。

**Q138：MTP 为什么可能有助于训练？**  
A：让隐藏状态承担比单步 next-token 更远一点的预测约束，可能改善表示或优化效率；但额外模块和损失权重要付成本，收益需看原论文消融。不能从第 59 页示意图推出“推理时必定每次输出多个 token”。

**Q139：MTP 与推理中的 speculative decoding/EAGLE 是同一个东西吗？**  
A：不是。MTP 首先是**训练目标/模型附加模块**；EAGLE 一类方案是用 draft 模型提出候选，再由目标模型验证以加速生成。MTP 模块可能为某些推理策略提供帮助，但“训练时预测多个未来 token”不自动等于部署了多 token 草稿接受流程。

**Q140：把 DeepSeek V3 的“便宜”全部归功于 MoE 有什么问题？**  
A：V3 同时使用 MoE、MLA、路由/通信工程、低精度与训练系统设计；MoE 主要影响专家容量与每 token FFN 计算，MLA 主要影响 KV cache/attention 访存，系统实现决定实际吞吐。需要分组件消融，不能单因归因。

## 11. 综合面试题与三个手算练习

**Q141：给出 $n=4096,d_k=d_v=64$，对比单头、无 softmax 情况的两种乘法顺序。**  
A：左结合的主乘法量约 $n^2(d_k+d_v)=4096^2\times128\approx2.15\times10^9$；右结合约 $2nd_kd_v=2\times4096\times64^2\approx3.36\times10^7$，账面约 64 倍差距。注意这只是同一**线性乘法式**的关联顺序，不是与 softmax attention 质量相同的 64 倍加速承诺。

**Q142：上述单头例子里，循环状态与直接 KV cache 分别有多少元素？**  
A：$S$ 有 $64\times64=4096$ 个元素；仅这一头的 K/V 历史有 $2\times4096\times64=524288$ 个元素，约为状态的 128 倍。实际模型仍有多层、多头、核化分母、门控或 full attention 层；这个对比不能当总显存数字。

**Q143：$T=1024,N=64,k=2$，capacity factor 为 $1.25$ 时，每专家容量如何估？**  
A：平均每专家分配 $kT/N=2\times1024/64=32$ 个 token，按简化式 $C\approx\lceil1.25\times32\rceil=40$。若某专家收到 50 个分配，则有 10 个超容量分配要 drop、重路由或由动态稀疏实现处理，具体取决于系统策略。

**Q144：假设共享非专家参数 1B，64 个 0.1B 的 routed 专家、每 token 选 4 个，不设 shared experts；怎么算？**  
A：总参数约 $1+64\times0.1=7.4$B；每 token 激活参数约 $1+4\times0.1=1.4$B。单卡存储不能只按 1.4B 算；每 token FLOPs 也不能用总参数 7.4B 直接套 dense 公式。

**Q145：一句话区别“线性注意力、动态稀疏注意力、MoE”。**  
A：线性注意力把历史压缩到递归状态；动态稀疏注意力显式选择少数历史位置做主 attention；MoE 从大量参数中选择少数 FFN 专家。前两者主要作用于**序列交互**，后者主要作用于**模型容量**。

**Q146：如果长文档问答质量下降，应该如何定位是状态压缩还是 router 问题？**  
A：先区分错误发生在使用循环/稀疏 attention 的位置检索环节，还是专家 FFN 的表达环节。固定 MoE，换回 full attention 或加入 full 层并测试“针在海里”等检索；固定 attention，比较 dense FFN 与 MoE 的路由统计、负载、任务表现。每次只变一类因素。

**Q147：如何向面试官解释“理论节省、显存节省、wall-clock 节省”三者关系？**  
A：理论 FLOPs 只计算运算量；显存取决于权重、激活、状态/KV 和优化器；wall-clock 还受 kernel、内存带宽、网络和负载不均影响。线性注意力可能省长序列配对与状态，MoE 可能省每 token 专家计算，但都不保证同等比例的端到端加速。

**Q148：一个模型用了 1/8 full 层、DSA、MoE，应如何逐层记账？**  
A：逐层列注意力类型与成本：full 层的 $O(n^2)$ 配对、DSA 的轻量索引和 $k$ 个主配对、线性层的固定状态更新；再列 FFN 是 dense 还是 MoE，分开算总/激活专家权重和通信。最后加 embedding、输出头、norm 与系统开销。不能只取三个“节省率”直接相乘。

**Q149：若论文声称“MoE 比 dense 训练快 2 倍”，优先问哪五件事？**  
A：①“快”是每步吞吐还是达到同等指标所需时间；②总/激活参数和每 token FLOPs；③训练 token、数据与任务；④ GPU/网络拓扑和 batch；⑤负载、丢 token 与质量是否相近。缺任一项都难判断收益来源和可迁移性。

**Q150：这节课最重要的设计原则是什么？**  
A：分别找出真正昂贵的维度，并把数学、表达能力与系统账本一起看：长上下文可压缩状态或稀疏读取，大模型容量可稀疏激活专家；但历史压缩可能损失精确回忆，稀疏路由带来均衡/通信问题，所以常见最有效方案是**有约束的混合架构加完整实测**。

## 12. 课件核对点、常见混淆与复习路线

| 容易说错的话 | 应如何修正 |
| --- | --- |
| “$Q(K^\top V)$ 就是 softmax attention 的等价加速。” | 仅当中间非线性被移除，或使用特定核化/近似形式时才能改写；因果情形须用前缀状态。 |
| “有固定比例 full attention 的混合模型严格线性。” | 固定数量的 dense full 层仍留下 $O(n^2)$ 项；实测近线性须说明长度区间和实现前提。 |
| “DSA 的总成本必定 $O(nk)$。” | 主 attention 是稀疏的；若 indexer 对所有历史位置评分，索引部分仍可能是 $O(n^2)$，只是常数较小。 |
| “MoE 每 token 只激活 37B，所以只存 37B 参数。” | `active` 更贴近一次计算路径；全部权重、优化器状态/分布式放置仍需单独核算。 |
| “top-$k$ 不可导，所以 router 完全没有梯度。” | 选中索引不可导，但门控权重/平衡统计中的连续概率通常仍能传梯度。 |
| “auxiliary-loss-free 就是完全没有辅助 loss。” | DeepSeek V3 主要用在线 bias 平衡，仍有小的 sequence-wise 辅助平衡项。 |
| “$\beta=0$ 时 Gated DeltaNet 状态原封不动。” | 无当前输入写入，但若 $\gamma\ne1$ 仍有遗忘。 |
| “MLA 只缓存 $c^{KV}$，没有别的东西。” | 解耦 RoPE 的位置分支仍需缓存相应的小 key 表示；具体布局看原实现。 |

**讲义中的两处笔误/简写：**

1. **第 56 页**标题为 DeepSeek MoE V3，却把 `671B / 37 active` 行写成 `V2`，并把 routed 专家写作 `258`。第 35 页表格与 [DeepSeek-V3 原报告](https://arxiv.org/abs/2412.19437)均给出 **256 routed、8 active、1 shared**；用原报告的明确配置复习。
2. **第 6 页**写 MiniMax-M1 “linear scaling in context length”，这应按其图表的具体测量条件理解；只要其中的 full attention 层仍对所有 $n$ 个位置做 dense 配对，混合网络的严格渐近上界并非纯 $O(n)$。

**建议自测顺序：**先不看答案，手推线性状态的形状与因果性 → 手算 top-$k$ 权重归一化 → 推 Switch 平衡 loss 的导数 → 画一遍 all-to-all 路由流程 → 区分 V3 的 MoE、MLA、MTP 三项作用。每道题尽量补上一句“何时这个结论不成立”。

## 参考资料

以下是课程讲义及与本讲主要技术对应的原始研究/机构报告。正文关于模型表现的陈述以讲义所展示的实验为范围，不能当成对所有模型的实时排行榜。

- [Stanford CS336 Spring 2026 Lecture 4 官方 PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_04.pdf)
- [Katharopoulos 等，Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)
- [Dao 与 Gu，Transformers are SSMs / Mamba-2](https://arxiv.org/abs/2405.21060)
- [Yang 等，Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)
- [MiniMax-M1 原报告](https://arxiv.org/abs/2506.13585)
- [NVIDIA Nemotron 3 Nano 技术报告](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf)
- [NVIDIA Nemotron 3 Super / LatentMoE 技术报告](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)
- [DeepSeek-V3.2：DeepSeek Sparse Attention](https://arxiv.org/abs/2512.02556)
- [Fedus 等，Switch Transformers](https://arxiv.org/abs/2101.03961)
- [Dai 等，DeepSeekMoE](https://arxiv.org/abs/2401.06066)
- [Muennighoff 等，OLMoE](https://arxiv.org/abs/2409.02060)
- [Gale 等，MegaBlocks](https://arxiv.org/abs/2211.15841)
- [Komatsuzaki 等，Sparse Upcycling](https://arxiv.org/abs/2212.05055)
- [DeepSeek-V2 原报告（MLA 与 MoE）](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 原报告（路由偏置、MLA、MTP）](https://arxiv.org/abs/2412.19437)

