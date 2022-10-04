import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('out_file', type=str)
parser.add_argument('model', type=str)
parser.add_argument('n_shard', type=int)
parser.add_argument('--print', type=int, default=1)
args = parser.parse_args()


