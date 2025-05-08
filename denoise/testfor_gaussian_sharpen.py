import cv2
import os
import argparse
import glob
import numpy as np
from utils import *  # 使用 batch_PSNR

parser = argparse.ArgumentParser(description="GaussianFilter+Sharpen_Test")
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
parser.add_argument("--kernel_size", type=int, default=3, help='gaussian filter kernel size')
opt = parser.parse_args()

def normalize(data):
    return data / 255.

def denormalize(data):
    return np.clip(data * 255., 0, 255).astype(np.uint8)

def add_titles(image_row, titles, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, thickness=1):
    h, w = image_row.shape[:2]
    section_width = w // 3
    title_height = 20
    new_image = np.zeros((h + title_height, w), dtype=np.uint8)
    new_image[title_height:, :] = image_row
    for i, title in enumerate(titles):
        text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
        x = section_width * i + (section_width - text_size[0]) // 2
        y = title_height - 5
        cv2.putText(new_image, title, (x, y), font, font_scale, 255, thickness, cv2.LINE_AA)
    return new_image

def main():
    print('Loading data ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()

    output_dir = f"demo_gaussian_sharpen{opt.kernel_size}x{opt.kernel_size}"
    os.makedirs(output_dir, exist_ok=True)

    psnr_total = 0
    for idx, f in enumerate(files_source, start=1):
        Img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        Img_norm = normalize(np.float32(Img))
        noise = np.random.normal(0, opt.test_noiseL / 255., Img_norm.shape)
        noisy_img = Img_norm + noise
        noisy_img = np.clip(noisy_img, 0., 1.)

        blurred = cv2.GaussianBlur(noisy_img, (opt.kernel_size, opt.kernel_size), 0)

        # 銳化處理
        sharpen_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
        sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)

        clean_img = denormalize(Img_norm)
        noisy_img_ = denormalize(noisy_img)
        denoised_img = denormalize(sharpened)

        # 計算 PSNR
        psnr = batch_PSNR(torch.Tensor(sharpened).unsqueeze(0).unsqueeze(0),
                          torch.Tensor(Img_norm).unsqueeze(0).unsqueeze(0), 1.)
        psnr_total += psnr

        # 存圖
        stacked = np.hstack((clean_img, noisy_img_, denoised_img))
        titled = add_titles(stacked, ['Original', 'Noisy', 'Gauss+Sharpen'])
        output_path = os.path.join(output_dir, f"{idx:04d}_compare.png")
        cv2.imwrite(output_path, titled)

        print(f"{f} PSNR {psnr:.6f}")

    psnr_avg = psnr_total / len(files_source)
    print(f"\nPSNR on test data: {psnr_avg:.6f}")

if __name__ == "__main__":
    main()
