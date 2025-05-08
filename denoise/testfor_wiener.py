import cv2
import os
import argparse
import glob
import numpy as np
from scipy.signal import wiener
from utils import *  # 使用 batch_PSNR

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

parser = argparse.ArgumentParser(description="Wiener_Filter_Test")
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
parser.add_argument("--kernel_size", type=int, default=3, help='Kernel size for Wiener filter (e.g., 3 or 7)')
opt = parser.parse_args()

def main():
    print('Loading data ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()

    folder_name = f"demo_wiener{opt.kernel_size}x{opt.kernel_size}"
    os.makedirs(folder_name, exist_ok=True)

    psnr_total = 0

    for idx, f in enumerate(files_source, start=1):
        Img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        Img_norm = normalize(np.float32(Img))
        noise = np.random.normal(0, opt.test_noiseL / 255., Img_norm.shape)
        noisy = Img_norm + noise
        noisy = np.clip(noisy, 0, 1)

        denoised = wiener(noisy, (opt.kernel_size, opt.kernel_size))
        denoised = np.clip(denoised, 0, 1)

        # PSNR 計算（統一格式：torch tensor）
        ISource = torch.tensor(Img_norm).unsqueeze(0).unsqueeze(0).cuda()
        IDenoised = torch.tensor(denoised).unsqueeze(0).unsqueeze(0).cuda()
        psnr = batch_PSNR(IDenoised, ISource, 1.)
        psnr_total += psnr

        print("%s PSNR %f" % (f, psnr))

        clean_img = denormalize(Img_norm)
        noisy_img = denormalize(noisy)
        denoised_img = denormalize(denoised)

        stacked = np.hstack((clean_img, noisy_img, denoised_img))
        titled = add_titles(stacked, ['Original', 'Noisy', f'Wiener {opt.kernel_size}x{opt.kernel_size}'])

        output_path = os.path.join(folder_name, f"{idx:04d}_compare.png")
        cv2.imwrite(output_path, titled)

    avg_psnr = psnr_total / len(files_source)
    print("\nPSNR on test data %f" % avg_psnr)

if __name__ == "__main__":
    main()
