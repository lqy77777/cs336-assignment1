import numpy as np
from cs336_basics.bpe import train_bpe,Tokenizer
from cs336_basics.tool import save_json,data_loading,tokenize_text_to_bin
from time import perf_counter

def main():
    #1.1训练bpe，得到vocab,merges和special_tokens
    input_path_1="data/TinyStoriesV2-GPT4-train.txt"
    input_path_2 = "data/TinyStoriesV2-GPT4-valid.txt"

    tokenizer_json_path = "data/tokenizer.json"

    train_path = 'data/train_data.bin'

    validation_path = 'data/validation_data.bin'

    special_tokens=["<|endoftext|>"]

    start = perf_counter()
    vocab, merges = train_bpe(
        input_path=input_path_1,
        vocab_size=10_000,
        special_tokens=special_tokens,
        num_processes=24
)
    save_json(vocab, merges, special_tokens,tokenizer_json_path)
    end = perf_counter()
    print(end-start)
if __name__ == '__main__':
    main()