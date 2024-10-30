import transformers
from transformers import AutoTokenizer

# Initialize the TinyStories tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
