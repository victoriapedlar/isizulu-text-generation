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

echo "---"
echo "Happy language modeling :)"