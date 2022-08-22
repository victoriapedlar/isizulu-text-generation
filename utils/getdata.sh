echo "=== Acquiring datasets ==="
echo "---"

export PYTHONPATH=$PYTHONPATH:`pwd`/utils

echo "- Downloading NCHLT data"
mkdir -p data/nchlt
mkdir -p data/nchlt/isizulu
python3 utils/get_nchlt.py --output_dir=data/nchlt

echo

echo "- Downloading autshumato data"
mkdir -p data/autshumato
mkdir -p data/autshumato/isizulu
python3 utils/get_autshumato.py --output_dir=data/autshumato

echo

echo "- Downloading isolezwe data"
mkdir -p data/isolezwe
mkdir -p data/isolezwe/isizulu
python3 utils/get_isolezwe.py --output_dir=data/isolezwe

echo

echo "- Partitioning datasets"
python3 utils/partition_datasets.py

echo

echo "- Concatenating datasets"
mkdir -p data/combined
mkdir -p data/combined/isizulu
cat data/isolezwe/isizulu/train.txt data/autshumato/isizulu/train.txt data/nchlt/isizulu/train.txt > data/combined/isizulu/train.txt
cat data/isolezwe/isizulu/valid.txt data/autshumato/isizulu/valid.txt data/nchlt/isizulu/valid.txt > data/combined/isizulu/valid.txt
cat data/isolezwe/isizulu/test.txt data/autshumato/isizulu/test.txt data/nchlt/isizulu/test.txt > data/combined/isizulu/test.txt

echo "---"
echo "Happy language modeling :)"