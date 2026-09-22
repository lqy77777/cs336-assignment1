# multiprocessing 并行与 BPE 预分词

本文结合当前 `cs336_basics/bpe.py`，解释如何把“读取分块 → 预分词 → 计数”分配给多个进程。示例用于学习，尚未写入项目代码；它们只覆盖预分词计数，不是完整的 BPE 训练实现。

## 1. multiprocessing 如何实现并行

普通 `for` 循环在当前进程中依次执行任务。仅设置 `num_processes = 4` 不会启动并行。

`multiprocessing` 可以创建多个工作进程，由操作系统调度它们使用 CPU。对于通常启用 GIL 的 CPython，多进程能够绕开单个解释器内 GIL 对 Python 代码执行的限制；不应据此认为所有多线程计算都无法并行。[Python 官方说明](https://docs.python.org/3.12/library/multiprocessing.html#introduction)

对于你的 BPE，任务可以这样分配：

```text
主进程：计算文件边界，构造任务列表
                 ↓
        进程池分配各个分块
                 ↓
进程 A：读分块 1 → 预分词 → Counter 1
进程 B：读分块 2 → 预分词 → Counter 2
进程 C：读分块 3 → 预分词 → Counter 3
                 ↓
主进程：接收结果，将局部 Counter 累加
                 ↓
主进程：根据全局频次继续进行 BPE 合并
```

并行的前提是任务之间能够独立处理。不同分块的预分词计数可以独立计算；BPE 各轮合并则依赖前一轮的结果，不能分别训练后直接拼接 `merges`。

## 2. 先认识进程池 Pool

进程池是一组可重复使用的工作进程。你提供“调用哪个函数”和“每次调用传什么参数”，进程池负责安排任务并传回结果。

| 名称 | 含义 |
| --- | --- |
| 主进程 | 组织任务、接收结果、执行汇总的进程 |
| 工作进程 | 执行具体任务的子进程 |
| `Pool(processes=4)` | 创建包含 4 个工作进程的进程池 |
| `pool.map(func, items)` | 将每个元素作为一个参数传给 `func` |
| `pool.starmap(func, tasks)` | 将每个任务元组展开为多个参数传给 `func` |

`map()` 和 `starmap()` 都等待结果，并按输入顺序返回结果列表；任务实际完成的先后顺序可以不同。进程池应通过 `with` 等方式管理资源。[进程池 API](https://docs.python.org/3.12/library/multiprocessing.html#multiprocessing.pool.Pool)

### 最小示例：并行计算乘积

先理解参数、变量与返回值：

- `multiply(a, b)`：工作函数，接收两个数字，返回它们的乘积。
- `tasks`：三个任务，每个任务是一个二元组。
- `pool`：包含两个工作进程的进程池。
- `results`：`starmap()` 返回的列表，内容为各任务返回值。
- `__name__`：模块运行方式的标识，直接运行脚本时为 `"__main__"`。

```python
from multiprocessing import Pool

def multiply(a, b):
    return a * b

if __name__ == "__main__":
    tasks = [(2, 3), (4, 5), (6, 7)]
    with Pool(processes=2) as pool:
        results = pool.starmap(multiply, tasks)
    print(results)
```

逐行解释：

1. 导入进程池。
2. 在模块顶层定义工作函数 `multiply()`。
3. `return a * b` 将计算结果交给调用方。
4. 判断是否直接运行当前脚本，避免模块被导入时启动任务。
5. 创建三个任务。
6. 创建两个工作进程。
7. 将三个任务交给进程池执行，收集结果。
8. 输出 `[6, 20, 42]`。

这里 `starmap()` 调用的效果相当于分别计算 `multiply(2, 3)`、`multiply(4, 5)` 和 `multiply(6, 7)`，并允许不同任务并行执行。

两个进程可以处理三个甚至更多任务：空闲的工作进程会继续领取工作。进程数量不必等于任务数量，也不能把任务固定理解为与某个进程一一对应。

## 3. BPE 工作函数：每次只处理一个分块

你当前循环体主要完成以下事情：

1. 定位并读取当前分块。
2. 按特殊 token 分隔文本。
3. 用 `PAT` 对各个普通文本片段预分词。
4. 将每个预分词编码成 UTF-8 字节串。
5. 累加它的出现次数。

将这些操作提取成一个函数，就得到了可以交给进程池的任务。

### 参数、变量与返回值

| 名称 | 含义 |
| --- | --- |
| `input_path` | 输入文件路径 |
| `start`、`end` | 分块的字节区间 `[start, end)` |
| `special_pattern` | 已转义并连接的特殊 token 正则；没有特殊 token 时使用 `None` |
| `PAT` | 模块顶层定义的预分词规则 |
| `f` | 当前工作进程自己打开的文件对象 |
| `chunk` | 当前分块解码后的字符串 |
| `parts` | 按特殊 token 分隔后的普通文本片段列表 |
| `pre_token` | 一次正则匹配的匹配对象 |
| `word` | 完整预分词对应的 UTF-8 字节串 |
| `frequency` | 当前分块独有的计数器 |
| 返回值 | `Counter[bytes]`，键是完整预分词，值是出现次数 |

### 示例代码

下面的导入、`PAT` 和函数定义均位于模块顶层。这里的 `re` 是第三方库 `regex` 的别名。

```python
from collections import Counter
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def count_chunk(
    input_path: str,
    start: int,
    end: int,
    special_pattern: str | None,
) -> Counter[bytes]:
    frequency = Counter()

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")

    parts = (
        re.split(special_pattern, chunk)
        if special_pattern is not None
        else [chunk]
    )

    for part in parts:
        for pre_token in re.finditer(PAT, part):
            word = pre_token.group().encode("utf-8")
            frequency[word] += 1

    return frequency
```

### 逐行解释函数体

1. `frequency = Counter()`：为这一次任务创建空计数器。
2. `with open(...) as f:`：工作进程自行打开输入文件。
3. `f.seek(start)`：移动到当前分块的起始字节位置。
4. `f.read(end - start).decode("utf-8")`：读取当前分块，并解码成文本。这里假定文件是合法 UTF-8，分块边界正确；解码失败会直接报错。
5. `parts = (...)`：有特殊 token 规则时切分文本，否则把整个分块作为一个片段。
6. `for part in parts:`：分别处理每个普通文本片段，不跨特殊 token 连接文本。
7. `re.finditer(PAT, part)`：依次产生正则匹配对象。
8. `.group().encode("utf-8")`：取出本次匹配的字符串并转成字节串。
9. `frequency[word] += 1`：完整预分词每出现一次，次数加一。
10. `return frequency`：将局部统计交给进程池传回主进程。

关键设计：函数内部维护局部计数器。它不依赖子进程修改主进程的全局变量。

## 4. 主进程：构造任务、调度和汇总

下面的函数接收已经计算好的 `boundaries`。它可以与上一节的 `count_chunk()` 放在同一个模块中；分块边界仍由你现有的 `find_chunk_boundaries()` 负责计算。

### 参数、变量与返回值

- `input_path`：输入文件路径。
- `boundaries`：递增的字节边界列表，包括 `0` 和文件大小。
- `special_tokens`：特殊 token 列表，元素应为非空字符串。
- `num_processes`：工作进程数量，必须大于零。
- `special_pattern`：普通文本切分规则，只需在主进程构造一次。
- `tasks`：工作任务列表，每个任务与 `count_chunk()` 的四个参数对应。
- `partial_counts`：各分块返回的计数器列表。
- `frequency`：汇总后的全局计数器，也是本函数的返回值。

### 示例代码

```python
from multiprocessing import Pool

def parallel_pretokenize(
    input_path: str,
    boundaries: list[int],
    special_tokens: list[str],
    num_processes: int = 4,
) -> Counter[bytes]:
    if num_processes < 1:
        raise ValueError("num_processes must be positive")
    if any(token == "" for token in special_tokens):
        raise ValueError("special tokens must not be empty strings")

    special_pattern = (
        "|".join(
            re.escape(token)
            for token in sorted(special_tokens, key=len, reverse=True)
        )
        if special_tokens
        else None
    )

    tasks = [
        (input_path, start, end, special_pattern)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if start < end
    ]

    frequency = Counter()
    if not tasks:
        return frequency

    with Pool(processes=num_processes) as pool:
        partial_counts = pool.starmap(count_chunk, tasks)

    for counts in partial_counts:
        frequency.update(counts)

    return frequency
```

### 按执行顺序解释

1. 检查进程数量和特殊 token 的基本有效性。
2. 将特殊 token 按长度降序排列，再逐个转义并用 `|` 连接。如果某个特殊 token 是另一个的前缀，这里约定优先匹配较长者。
3. 特殊 token 列表为空时使用 `None`，避免用空正则在字符之间切分文本。
4. 通过相邻边界构造 `(start, end)`，并加入文件路径和分隔规则。
5. 创建用于汇总的空 `Counter`。
6. 没有非空分块时直接返回空计数器。
7. 创建进程池。
8. `starmap(count_chunk, tasks)` 将每个元组展开为四个参数，调用工作函数并收集结果。这里传的是函数 `count_chunk`，不是先执行 `count_chunk(...)`。
9. 遍历各任务的返回值，使用 `Counter.update()` 累加。
10. 返回全局频次，供后续 BPE 训练使用。

注意：`Counter.update()` 原地累加，返回 `None`。不要写 `frequency = frequency.update(counts)`；也不要把普通字典的覆盖式 `update()` 与它混淆。

例如：

```text
进程 A 返回：{b"hello": 2, b" world": 1}
进程 B 返回：{b"hello": 3, b"python": 1}

汇总结果：   {b"hello": 5, b" world": 1, b"python": 1}
```

这是“局部统计 → 全局相加”的计算方式。无论两个分块谁先完成，正整数频次相加的结果都相同。

## 5. 如何接到你现有的训练函数中

保留主进程的边界计算，把原来的串行预分词循环替换为对 `parallel_pretokenize()` 的调用即可。以下为衔接片段，需要放在已有训练函数内。

参数和返回值：`input_path`、`special_tokens` 来自训练函数参数；`num_processes` 是工作进程数；`boundaries` 是边界列表；`frequency` 接收全局预分词频次。

```python
with open(input_path, "rb") as f:
    boundaries = find_chunk_boundaries(
        f, num_processes, b"<|endoftext|>"
    )

frequency = parallel_pretokenize(
    input_path, boundaries, special_tokens, num_processes
)
```

按顺序解释：先打开文件计算边界，然后退出 `with` 关闭主进程中的文件，最后启动并行预分词并得到汇总结果。

这里沿用了你当前的边界标记。它假设 `<|endoftext|>` 是语料中可用的文档分隔符，并且包含在需要排除的特殊 token 中。不能对任意语料都硬编码这个假设；没有这样的标记时，需要另外设计能保留预分词语义的分块策略。

`frequency` 不是最终的 `vocab` 或 `merges`。例如 `b"hello"` 出现 5 次，后续初始化为单字节 token 序列后，其中每次相邻 pair 出现都应按频次 5 加权统计。

## 6. 为什么需要 main 保护和顶层函数

在 macOS 上，通常使用 `spawn` 启动子进程：启动新的解释器，再导入相关模块。脚本调用入口应放在 `if __name__ == "__main__":` 下，工作函数则定义在模块顶层，方便子进程导入。任务参数和返回值需要能够序列化。[启动方式](https://docs.python.org/3.12/library/multiprocessing.html#contexts-and-start-methods)、[安全导入主模块](https://docs.python.org/3.12/library/multiprocessing.html#safe-importing-of-main-module)

实际组织时：

- 模块顶层：导入语句、`PAT`、`count_chunk()`、`parallel_pretokenize()`、`train_bpe()` 的定义。
- main 保护内部：脚本实际启动训练的调用，以及训练结果的打印或保存。
- 如果训练函数由测试框架调用，不必把训练函数定义放进 main 保护；重点是模块导入时不要自动运行训练。

文件路径、整数、字符串、由字节键和整数值组成的 `Counter` 都适合这里的进程间传输。不要传入打开的文件对象 `f`、正则匹配迭代器或嵌套的工作函数。

练习进程池示例时，优先在可导入的普通 Python 脚本中运行，避免交互环境对工作函数导入带来的干扰。

## 7. 为什么不用多个进程共同修改一个 Counter

普通 Python 对象不会因为创建了多个进程就自动成为共享对象。子进程修改自己持有的普通 `Counter`，不会自动更新主进程中的同名变量。

对于你的预分词，推荐每个任务独立计数，完成后返回，再由主进程累加。这样无需为每次 `frequency[word] += 1` 进行跨进程通信。

如果使用共享代理字典，每个预分词都远程读写会增加通信成本，而且“读取旧值 → 加一 → 写回”是复合操作，需要另行处理同步。当前任务没有必要引入这套机制。

## 8. 分块正确性比启动进程更重要

多进程不会自动保证分块正确。串行和并行都应满足：

- 分块没有重叠或遗漏，覆盖整个文件。
- 边界落在允许独立处理的位置，不能截断 UTF-8 字符或普通预分词。
- 特殊 token 用于分隔文本，各片段独立预分词，不能删除标记后把两侧文本拼接起来。
- 特殊 token 不进入普通预分词计数，但后续仍需按训练要求加入词表。

你目前的 `find_chunk_boundaries()` 以 4096 字节窗口向后搜索标记。一个值得检查的边界情况是：特殊 token 恰好横跨两个读取窗口时，单独对每个窗口调用 `find()` 会漏掉它。改进方向是保留前一个窗口末尾的部分字节，与下一窗口共同搜索，并正确计算绝对偏移。

此外，边界去重后可能少于期望分块数。如果只有一个有效分块，即使创建四个工作进程，这个分块仍只能作为一个任务执行。

## 9. 并行不保证更快

进程启动、任务传输、局部结果传回以及主进程汇总都需要时间。小文件可能串行更快。

可以按以下顺序优化：

1. 先确认单个分块的处理结果正确。
2. 接入多进程，确认全局计数没有变化。
3. 用同一份语料比较串行与 2、4 个工作进程的总耗时。
4. 检查是否存在特别大的分块，让一个进程拖慢整体完成时间。
5. 数据量大时，关注同时保存各块文本、局部计数器和结果列表带来的内存占用。

文件分块和进程池的任务批次是两个概念：文件边界决定一项任务读取哪些字节；进程池 API 中的 `chunksize` 决定一次调度打包多少项任务，并不会修改文件的切分边界。

`starmap()` 便于入门，但会集中返回结果列表。若后续内存成为问题，可考虑使用 `imap_unordered()` 配合接收单个任务元组的顶层包装函数，让主进程逐个消费返回结果并累加；这仍不意味着完全没有内部缓冲。

## 10. 如何验证理解和实现

### 验证结果

先对一份小语料比较“同样分块的串行处理”和“并行处理”。若结果不同，重点检查任务参数传递和计数汇总。

再比较“整份文本一次处理”和“分块处理”。若前一项一致而这一项不同，重点检查文件边界、特殊 token 和 UTF-8 解码。

测试语料可包含：重复单词、中文、换行、多个特殊 token、连续特殊 token，以及没有特殊 token 的情况。语料应使用相同的预分词规则和特殊 token 策略。

### 自测题

1. 为什么 `num_processes = 4` 不能让普通 `for` 循环自动并行？
2. 为什么 `count_chunk()` 要自己打开文件，并创建自己的 `Counter`？
3. `pool.starmap(count_chunk, tasks)` 中，任务元组的顺序应与什么对应？
4. 为什么要写 `return frequency`，而不能只修改一个全局变量？
5. 两个分块分别返回 `hello: 2` 和 `hello: 3`，汇总时应覆盖还是累加？
6. 为什么不能让每个分块独立训练 BPE 后直接拼接合并表？

<details>
<summary>参考答案</summary>

1. 变量赋值没有创建进程，需要显式使用进程池等工具调度任务。
2. 每个任务独立管理文件读取位置和局部计数，任务之间不需要同步这些状态。
3. 与 `count_chunk(input_path, start, end, special_pattern)` 的参数顺序对应。
4. 普通全局变量不自动跨进程共享，返回值才会经进程池传回主进程。
5. 累加，得到 `hello: 5`，使用 `Counter.update()`。
6. 每轮 BPE 合并需要依据全局频次，而且下一轮依赖上一轮的结果。

</details>
