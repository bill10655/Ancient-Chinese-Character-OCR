# %%
from PIL import Image
import cv2
import argparse
import numpy as np
from pathlib import Path


#%%
parser = argparse.ArgumentParser(description="處理被 split.py 切割出來的檔案")
parser.add_argument("input_dir", type=str, help="要被處理的資料夾路徑", default="split_results")
parser.add_argument("output_dir", type=str, help="輸出結果的資料夾路徑", default="clean_results")
args = parser.parse_args()


#%%
if __name__ == "__main__":
    base_dir = Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)
    fcount = len(list(input_dir.iterdir()))
    for idx1, folder in enumerate(input_dir.iterdir()):
        out_dir = base_dir / Path(folder.stem)
        out_dir.mkdir(parents=True, exist_ok=True)

        # log
        if idx1 % 100 == 0:
            print(f"{idx1}/{fcount}")

        for idx, fname in enumerate(folder.glob("*.png")):
            img = cv2.imread(str(fname), cv2.IMREAD_COLOR)
            
            # If the size of image is too small then skip the image
            if img.shape[0] < 15 or img.shape[1] < 20:
                continue
            
            # If ratio is too small then skip the image
            ratio = np.sum(img) / (img.shape[0] * img.shape[1])
            if ratio < 70:
                continue
            
            # Trans to Gray Image
            gray_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 3, 3, 3)
            coefficients = [0, 1, 1]
            m = np.array(coefficients).reshape((1, 3))
            gray_img = cv2.transform(gray_img, m)
            
            # Binary
            ret, binary_img = cv2.threshold(gray_img, 180, 255, cv2.THRESH_BINARY)
            
            # Morpholgy
            ele = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
            ele2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morphology_img = cv2.morphologyEx(
                binary_img, cv2.MORPH_OPEN, ele, iterations=1
            )
            dilation_img = cv2.dilate(morphology_img, ele2, iterations=5)
            canny_img = cv2.Canny(morphology_img, 50, 100)
            canny_dilation_img = cv2.Canny(dilation_img, 50, 100)
            
            # Contours (Bounding Box)
            contour_img = dilation_img.copy()
            
            # Canny
            _, cnts, _ = cv2.findContours(
                contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if cnts == []:
                continue
            areas = [cv2.contourArea(c) for c in cnts]
            i = np.argmax(areas)
            x, y, w, h = cv2.boundingRect(cnts[i])

            offset = 4
            x -= offset
            if x < 0:
                x = 0
            y -= offset
            if y < 0:
                y = 0
            final_img = img[y : y + h, x : x + w]
            height, width, channel = final_img.shape
            if height <= 6 or width <= 30 or channel <= 0:
                continue
            cv2.imwrite(f"{out_dir/Path(fname.stem)}.png", final_img)
