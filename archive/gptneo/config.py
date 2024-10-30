from dataclasses import dataclass

@dataclass
class TinyStories33MConfig: # 33,019,776 parameters
    block_size: int = 2048 
    vocab_size: int = 50257
    n_layer: int = 4
    n_head: int = 3
    n_embd: int = 768

@dataclass
class TinyStories8MConfig: # 8,194,048 parameters
    block_size: int = 2048
    vocab_size: int = 50257
    n_layer: int = 8
    n_head: int = 3
    n_embd: int = 256


@dataclass
class TinyStories1MConfig: # 1,048,576 parameters
    block_size: int = 2048
    vocab_size: int = 50257
    n_layer: int = 8
    n_head: int = 3
    n_embd: int = 64