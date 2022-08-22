import re
import argparse
import requests
import zipfile
import os
from io import BytesIO
import utils

# take --output_dir command-line argument
parser = argparse.ArgumentParser(description="Download NCHLT datasets.")
parser.add_argument(
    "--output_dir", required=True, help="directory where output files will be saved"
)
args = parser.parse_args()

datasets = [
    (
        "http://www.rma.nwu.ac.za/bitstream/handle/20.500.12185/321/corpora.nchlt.zu.zip",  # url
        "zu/2.Corpora/CORP.NCHLT.zu.CLEAN.2.0.txt",  # file_name
        "isizulu/isizulu.txt",  # output_name
        11,  # lines_to_remove
    ),
]

for url, file_name, output_name, lines_to_remove in datasets:
    print("processing:", url)

    r = requests.get(url)
    zip = zipfile.ZipFile(BytesIO(r.content))
    corpus = zip.open(file_name).read().decode("utf-8").strip()

    # remove tags containing article filenames
    corpus = re.sub(r"<fn>.*</fn>", "", corpus)

    # put each sentence on a new line
    corpus = corpus.replace(". ", ".\n")

    # remove empty lines from corpus
    sentences = corpus.splitlines()

    sentences = utils.clean_sentences(
        sentences,
        illegal_substrings=["\ufeff", "="],
        lines_to_remove=lines_to_remove,
    )

    # write article to file (with each sentence on a new line)
    output_file_name = os.path.join(args.output_dir, output_name)
    with open(output_file_name, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences))

    print("total sentences in {}:".format(output_name), corpus.count("\n"))
