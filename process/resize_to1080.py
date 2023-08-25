import argparse
import os
import subprocess
from tqdm import tqdm
import cv2

def resize(input_folder,output_folder ):
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # 构建图像文件的完整路径
            img_path = os.path.join(input_folder, filename)

            # 读取图像
            image = cv2.imread(img_path)

            # 调整图像大小
            resized_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            # 构建输出图像文件的完整路径
            output_path = os.path.join(output_folder, filename)

            # 保存调整后的图像
            cv2.imwrite(output_path, resized_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--picture_folder", type=str, default="/mnt/users/chenmuyin/chaofen/RT4KSR-main/res/0707/train1")
    parser.add_argument("--output_folder", type=str, default="/mnt/users/chenmuyin/chaofen/RT4KSR-main/res/0707/train1/area")
    args = parser.parse_args()


    picture_folder = args.picture_folder
    output_folder = args.output_folder
    resize(picture_folder,output_folder )