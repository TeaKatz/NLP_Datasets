''' Script for downloading all GLUE data.

Note: for legal reasons, we are unable to host MRPC.
You can either use the version hosted by the SentEval team, which is already tokenized,
or you can download the original data from (https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B-3604ED519838/MSRParaphraseCorpus.msi) and extract the data from it manually.
For Windows users, you can run the .msi file. For Mac and Linux users, consider an external library such as 'cabextract' (see below for an example).
You should then rename and place specific files in a folder (see below for an example).

mkdir MRPC
cabextract MSRParaphraseCorpus.msi -d MRPC
cat MRPC/_2DEC3DBE877E4DB192D17C0256E90F1D | tr -d $'\r' > MRPC/msr_paraphrase_train.txt
cat MRPC/_D7B391F9EAFF4B1B8BCE8F21B20B1B61 | tr -d $'\r' > MRPC/msr_paraphrase_test.txt
rm MRPC/_*
rm MSRParaphraseCorpus.msi
'''

import os
import sys
import shutil
import argparse
import tempfile
import urllib
import io
if sys.version_info >= (3, 0):
    import urllib.request
import zipfile
from ..config import BASE_DIR, GLUE
from ..utilities import download_file_from_google_drive

URLLIB=urllib
if sys.version_info >= (3, 0):
    URLLIB=urllib.request


def download_and_extract(task, data_dir):
    print("Downloading and extracting %s..." % task)
    if task == "MNLI":
        print("\tNote (12/10/20): This script no longer downloads SNLI. You will need to manually download and format the data to use SNLI.")
    data_file = "%s.zip" % task
    URLLIB.urlretrieve(GLUE.TASK2PATH[task], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)
    print("\tCompleted!")

def format_mrpc(data_dir, path_to_data):
    print("Processing MRPC...")
    mrpc_dir = os.path.join(data_dir, "MRPC")
    if not os.path.isdir(mrpc_dir):
        os.mkdir(mrpc_dir)
    if path_to_data:
        mrpc_train_file = os.path.join(path_to_data, "msr_paraphrase_train.txt")
        mrpc_test_file = os.path.join(path_to_data, "msr_paraphrase_test.txt")
    else:
        try:
            mrpc_train_file = os.path.join(mrpc_dir, "msr_paraphrase_train.txt")
            mrpc_test_file = os.path.join(mrpc_dir, "msr_paraphrase_test.txt")
            URLLIB.urlretrieve(GLUE.MRPC_TRAIN, mrpc_train_file)
            URLLIB.urlretrieve(GLUE.MRPC_TEST, mrpc_test_file)
        except urllib.error.HTTPError:
            print("Error downloading MRPC")
            return
    assert os.path.isfile(mrpc_train_file), "Train data not found at %s" % mrpc_train_file
    assert os.path.isfile(mrpc_test_file), "Test data not found at %s" % mrpc_test_file

    with io.open(mrpc_test_file, encoding='utf-8') as data_fh, \
            io.open(os.path.join(mrpc_dir, "test.tsv"), 'w', encoding='utf-8') as test_fh:
        header = data_fh.readline()
        test_fh.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for idx, row in enumerate(data_fh):
            label, id1, id2, s1, s2 = row.strip().split('\t')
            test_fh.write("%d\t%s\t%s\t%s\t%s\n" % (idx, id1, id2, s1, s2))

    try:
        URLLIB.urlretrieve(GLUE.TASK2PATH["MRPC"], os.path.join(mrpc_dir, "dev_ids.tsv"))
    except KeyError or urllib.error.HTTPError:
        print("\tError downloading standard development IDs for MRPC. You will need to manually split your data.")
        print(f"\tDownloading train.tsv: {GLUE.MRPC_TRAIN_URL}")
        download_file_from_google_drive(GLUE.MRPC_TRAIN_ID, GLUE.MRPC_PATH + "/train.tsv")
        print(f"\tDownloading dev.tsv: {GLUE.MRPC_DEV_URL}")
        download_file_from_google_drive(GLUE.MRPC_DEV_ID, GLUE.MRPC_PATH + "/dev.tsv")
        return

    dev_ids = []
    with io.open(os.path.join(mrpc_dir, "dev_ids.tsv"), encoding='utf-8') as ids_fh:
        for row in ids_fh:
            dev_ids.append(row.strip().split('\t'))

    with io.open(mrpc_train_file, encoding='utf-8') as data_fh, \
         io.open(os.path.join(mrpc_dir, "train.tsv"), 'w', encoding='utf-8') as train_fh, \
         io.open(os.path.join(mrpc_dir, "dev.tsv"), 'w', encoding='utf-8') as dev_fh:
        header = data_fh.readline()
        train_fh.write(header)
        dev_fh.write(header)
        for row in data_fh:
            label, id1, id2, s1, s2 = row.strip().split('\t')
            if [id1, id2] in dev_ids:
                dev_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
            else:
                train_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))

    print("\tCompleted!")

def download_diagnostic(data_dir):
    print("Downloading and extracting diagnostic...")
    if not os.path.isdir(os.path.join(data_dir, "diagnostic")):
        os.mkdir(os.path.join(data_dir, "diagnostic"))
    data_file = os.path.join(data_dir, "diagnostic", "diagnostic.tsv")
    URLLIB.urlretrieve(GLUE.TASK2PATH["diagnostic"], data_file)
    print("\tCompleted!")
    return

def download_glue():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
        
    if os.path.exists(GLUE.PATH):
        return

    for task in GLUE.TASKS:
        if task == "MRPC":
            format_mrpc(GLUE.PATH, "")
        elif task == "diagnostic":
            download_diagnostic(GLUE.PATH)
        else:
            download_and_extract(task, GLUE.PATH)

    # Move diagnostic.tsv to MNLI
    shutil.move(GLUE.PATH + "/diagnostic/diagnostic.tsv", GLUE.PATH + "/MNLI/diagnostic.tsv")
    shutil.rmtree(GLUE.PATH + "/diagnostic")