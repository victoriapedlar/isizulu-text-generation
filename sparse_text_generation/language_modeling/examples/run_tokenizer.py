import os
from tokenizers import ByteLevelBPETokenizer
from ..pytorch_transformers import GPT2Tokenizer

# Set the dataset paths
paths = [
    "data/test/test.txt",
    "data/test/train.txt",
    "data/test/valid.txt",
]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(
    files=paths,
    vocab_size=8000,
    special_tokens=["<|endoftext|>"],
    show_progress=False,
)

# Create the output directory if it doesn't exist
output_dir = "./tokenizers/ByteLevelBPETokenizer/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the trained tokenizer
tokenizer.save_model(output_dir)
