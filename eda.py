import os


def count_words(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        words = text.split()
        return len(words)


corpuses = [
    "autshumato/isizulu",
    "isolezwe/isizulu",
    "nchlt/isizulu",
    "combined/isizulu",
]
file_types = ["train.txt", "valid.txt", "test.txt"]
base_folder = "data"

for corpus in corpuses:
    print(f"Word counts for {corpus}:")
    for file_type in file_types:
        file_path = os.path.join(base_folder, corpus, file_type)
        word_count = count_words(file_path)
        print(f"  {file_type}: {word_count}")
    print()
