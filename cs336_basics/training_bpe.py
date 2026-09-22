
import numpy as np
from cs336_basics.bpe import train_bpe,Tokenizer
from cs336_basics.tool import save_json,data_loading,tokenize_text_to_bin

def main():
    #1.1训练bpe，得到vocab,merges和special_tokens
    input_path="data/TinyStoriesV2-GPT4-train.txt"
    input_path_2 = ""
    tokenizer_json_path = "data/tokenizer.json"
    train_path = 'data/train_data.bin'
    validation_path = 'data/validation_data.bin'

    special_tokens=["<|endoftext|>"]
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=10_000,
        special_tokens=special_tokens,
)
    #1.2将vocab,merges和special_tokens保存到json文件中，并生成对应的tokenizer
    save_json(vocab, merges, special_tokens,tokenizer_json_path)
    tokenizer = Tokenizer(
        vocab = vocab,
        merges = merges,
        special_tokens= special_tokens
    )
    #2.1将.txt文本转换为token id(用bin方法：.bin 方法的核心是：
    #每次读取一小段文本，立刻编码成 token IDs，然后把这些整数以二进制形式追加到磁盘，不在内存中保存完整 token 数组。)
    tokenize_text_to_bin(tokenizer,input_path,train_path)
    tokenize_text_to_bin(tokenizer,input_path_2,validation_path)
if __name__ == '__main__':
    main()