#%%
import re
import cv2
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt


#%%
with open("RAWDFZData.txt", "r", encoding="utf-8") as infile:
    data = infile.read()

clean_data = data.replace("○", "")
pages = re.split(r"<", clean_data)


# %%
results = {}

for page in pages[1:]:
    try:
        tag, content = re.split(r">", page)
        b = re.search(r'b="([0-9]*?)"', tag).group(1)
        p = re.search(r'p="([0-9]*?)"', tag).group(1)
        fname = f"{b}_{p}"
        results[fname] = content
    except:
        pass


#%%
parser = argparse.ArgumentParser(description="把文字圖片檔跟文字做結合")
parser.add_argument("input_dir", type=str, help="文字圖片檔的資料夾路徑", default="clean_results")
parser.add_argument(
    "output_dir", type=str, help="輸出結果的資料夾路徑", default="combined_results"
)
args = parser.parse_args()


#%%
base_dir = Path(args.output_dir)
base_dir.mkdir(parents=True, exist_ok=True)


#%%
for folder in Path(args.input_dir).iterdir():
    out_dir = base_dir / Path(folder.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(folder)

    fnames = [str(fname) for fname in folder.glob("*.png")]
    sorted_fnames = sorted(fnames, key=lambda x: int(x.split("_")[-1][:-4]))
    results[folder.stem] = list(results[folder.stem])

    try:
        results[folder.stem].remove("一")
    except ValueError:
        pass

    for idx, (fname, word) in enumerate(zip(sorted_fnames, results[folder.stem])):
        img = cv2.imread(str(fname), cv2.IMREAD_COLOR)
        cv2.imwrite(f"{out_dir/Path(f'{Path(fname).stem}_{word}')}.png", img)
