#%%
import os
import cv2
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


#%%
parser = argparse.ArgumentParser(description="對地方志的原檔做一些簡單的處理")
parser.add_argument("input_dir", type=str, help="地方志圖片檔資料夾路徑", default="data/images")
parser.add_argument("output_dir", type=str, help="輸出結果的資料夾路徑", default="preprocessing_images")
args = parser.parse_args()


#%%
input_path = Path(args.input_dir)
output_path = Path(args.output_dir)
output_path.mkdir(parents=True, exist_ok=True)


# %%
def remain_edge(li, size):
    if li == []:
        return (0, 0)

    begin, end = 0, 0
    for element in li:
        if element < 30:
            begin = element
        elif element > size[1] - 35:
            end = element
            break
    return (begin, end)


# %%
idx = 0
for fname in input_path.glob("*.png"):
    img = cv2.imread(str(fname), cv2.IMREAD_GRAYSCALE)

    for row in range(img.shape[0]):
        if sum(img[row]) < 100000 and (row < 50 or row > img.shape[0] - 50):
            img[row] = 255

    img[:, img.shape[1] - 20 :] = 255
    img[:, :20] = 255
    img[:20] = 255
    img[img.shape[0] - 20 :] = 255

    cv2.imwrite(f"{output_path/fname.stem}.png", img)
    print(f"{output_path/fname.stem}.png")
