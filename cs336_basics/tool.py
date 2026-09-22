import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int
import os
import json
from typing import BinaryIO,IO
from pathlib import Path
from cs336_basics.bpe import Tokenizer
from optimization import cross_entropy

def data_loading(
        token_ids: Int[np.ndarray,"num_tokens"],
        batch_size: int,
        seq_len: int,
        device = None
) -> tuple[Int[Tensor,"batch_size seq_len"], Int[Tensor,"batch_size seq_len"]]:
    """data_loading 输入使用 NumPy array，主要是因为这里的 dataset 代表完整的 token 数据集，
    它可能非常大；PyTorch Tensor 则更适合作为已经采样出来、即将送进模型的小 batch。"""
    """
    随机在整个token_ids上采样batch_size次，每次的长度seq_len
    """
    legal_start = len(token_ids) - seq_len
    start_indices = np.random.randint(low = 0, high = legal_start,size = batch_size)
    off_set = np.arange(stop = seq_len)
    inputs_np = token_ids[start_indices[:,None] + off_set[None,:]]
    targets_np = token_ids[start_indices[:,None] + off_set[None,:] + 1]
    inputs = torch.tensor(inputs_np, dtype = torch.long,device = device)   #这里是小写工厂函数
    targets = torch.tensor(targets_np, dtype = torch.long,device = device)
    return inputs, targets

def save_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes]
):
    state_model = model.state_dict()
    state_opt = optimizer.state_dict()
    obj = {'model': state_model,'opt': state_opt,'iteration': iteration}
    torch.save(obj,out)

def load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: nn.Module,
        optimizer: torch.optim.Optimizer
):
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['opt'])
    return obj['iteration']

def save_json(
        vocab: dict[int,bytes],
        merges: list[tuple[bytes,bytes]],
        special_tokens: list[str],
        output: str | os.PathLike | BinaryIO | IO[bytes]
) -> None:
    #把vocab,merges,special_tokens保存为json文件，
    #由于json文件无法保存bytes，所以要把bytes转化为hex保存进文件中
    #JSON object 的键只能是字符串，不能真正保存整数键
    output = Path(output)
    data = {
        "vocab":{str(token_id): token_bytes.hex() for token_id, token_bytes in vocab.items()},
        "merges":[(left.hex(),right.hex())for left,right in merges],
        "special_tokens": special_tokens
    }
    with output.open(mode="w",encoding = "utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent = 2)
        f.write("\n")
    return

def tokenize_text_to_bin(
        tokenizer: Tokenizer,
        input_path: str | os.PathLike,
        output_path: str |os.PathLike,
        dtype: np.dtype = None
) -> int:
    """
    返回值：写入的token_总数
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    num_tokens = 0
    with (input_path.open(mode = 'r', encoding = 'utf-8') as input_file,output_path.open(mode = 'wb') as output_file):
        for line_number, text in enumerate(input_file,start = 1):
            token_ids = tokenizer.encode(text)
            token_array = np.asarray(token_ids,dtype = dtype)
            token_array.tofile(output_file)
            num_tokens += token_array.size
    return num_tokens

def make_fixed_batches(
        data: Int[np.ndarray,"num_tokens"], #通常是验证集memmap
        batch_size: int,
        context_length: int,
        device: torch.device,
        num_batches: int,
        seed: int,
) -> list[tuple[Tensor, Tensor]]:
    """提前从验证集随机采样固定的一组 batch。
    之后每次验证都使用同样的数据，避免验证曲线因为每次随机样本不同而抖动。
    """
    state = np.random.get_state()
    np.random.seed(seed)
    batches = [data_loading(data, batch_size, context_length, device) for _ in range(num_batches)]
    np.random.set_state(state)
    return batches

def log_jsonl(path: str, record: dict) -> None:
    """每条日志追加一行 JSON。Section 6/7 画曲线时直接读这个文件，不必去 grep stdout。"""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

@torch.no_grad()
def evaluate(model: nn.Module, batches: list[tuple[Tensor, Tensor]]) -> float:
    """在固定验证 batch 上计算平均交叉熵，并禁止构建反向传播计算图"""
    model.eval()
    total = 0.0
    for inputs, targets in batches:
        total += cross_entropy(model(inputs), targets).item()
    model.train()
    return total / len(batches)