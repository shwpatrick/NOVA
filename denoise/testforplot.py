import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
opt = parser.parse_args()

def normalize(data):
    return data / 255.

def denormalize(data):
    return np.clip(data * 255., 0, 255).astype(np.uint8)

def add_titles(image_row, titles, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, thickness=1):
    h, w = image_row.shape[:2]
    section_width = w // 3
    title_height = 20
    # 建立新圖像（加高）
    new_image = np.zeros((h + title_height, w), dtype=np.uint8)
    new_image[title_height:, :] = image_row

    for i, title in enumerate(titles):
        text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
        x = section_width * i + (section_width - text_size[0]) // 2
        y = title_height - 5
        cv2.putText(new_image, title, (x, y), font, font_scale, 255, thickness, cv2.LINE_AA)
    return new_image

def main():
    # Build model
    print('Loading model ...\n')
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    model = nn.DataParallel(net, device_ids=[0]).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    model.eval()

    # Load data
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()

    os.makedirs("demo_1epoch", exist_ok=True)

    psnr_test = 0
    for idx, f in enumerate(files_source, start=1):
        # 讀圖
        Img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        Img_norm = normalize(np.float32(Img))
        Img_input = np.expand_dims(Img_norm, 0)
        Img_input = np.expand_dims(Img_input, 0)
        ISource = torch.Tensor(Img_input)

        # 加噪
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
        INoisy = ISource + noise
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())

        # 推論
        with torch.no_grad():
            Out = torch.clamp(INoisy - model(INoisy), 0., 1.)
        psnr = batch_PSNR(Out, ISource, 1.)
        psnr_test += psnr
        print("%s PSNR %f" % (f, psnr))

        # 圖片轉回 numpy 格式
        clean_img = denormalize(ISource.cpu().numpy().squeeze())
        noisy_img = denormalize(INoisy.cpu().numpy().squeeze())
        denoised_img = denormalize(Out.cpu().numpy().squeeze())

        # 合併圖像 + 標題
        stacked = np.hstack((clean_img, noisy_img, denoised_img))
        titled = add_titles(stacked, ['Original', 'Noisy', 'Denoised'])

        output_path = os.path.join("demo_1epoch", f"{idx:04d}_compare.png")
        cv2.imwrite(output_path, titled)

    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)

if __name__ == "__main__":
    main()
