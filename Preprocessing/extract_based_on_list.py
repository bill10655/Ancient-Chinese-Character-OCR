#%%
import pickle
import argparse
from pathlib import Path
from shutil import copyfile
from distutils.dir_util import copy_tree

with open("data1_list.pickle", "rb") as infile:
    li = pickle.load(infile)


#%%
parser = argparse.ArgumentParser(description="根據給定的 list 挑出特定的資料夾出來")
parser.add_argument("input_dir", type=str, help="文字圖片檔的資料夾路徑", default="combined_results")
parser.add_argument(
    "output_dir", type=str, help="輸出結果的資料夾路徑", default="combined_results_data1"
)
args = parser.parse_args()


#%%
base_dir = Path(args.input_dir)
out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    for fname in li:
        from_folder = base_dir / Path(fname)
        out_folder = out_dir / Path(fname)
        copy_tree(str(from_folder), str(out_folder))