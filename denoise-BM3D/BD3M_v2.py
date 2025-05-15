
import os
import cv2
import time
import sys
from scipy.fftpack import dct, idct
import numpy as np
import glob
import matplotlib.pyplot as plt
import argparse
import torch
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def add_titles(image_row, titles, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, thickness=1):
    """
    在拼接好的橫向圖像上方加入對應文字標題

    Args:
        image_row (np.ndarray): 已經 np.hstack 完成的橫向圖像 (灰階)
        titles (list): 與圖像數量對應的標題清單
        font: OpenCV 字體
        font_scale: 字體大小
        thickness: 線條粗細

    Returns:
        np.ndarray: 上方加上文字的圖像
    """
    h, w = image_row.shape[:2]
    num_sections = len(titles)
    section_width = w // num_sections
    title_height = 20

    new_image = np.zeros((h + title_height, w), dtype=np.uint8)
    new_image[title_height:, :] = image_row

    for i, title in enumerate(titles):
        text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
        x = section_width * i + (section_width - text_size[0]) // 2
        y = title_height - 5
        cv2.putText(new_image, title, (x, y), font, font_scale, 255, thickness, cv2.LINE_AA)

    return new_image

def batch_PSNR(img, imclean, data_range):
    """
    計算PSNR方法
    原版的是正則化後進來
    現用的方法沒有正則化，所以在內部除了255

    :param img: 圖片來源
    :param imclean: 濾雜訊的圖片
    :param data_range: 圖片數量
    :return: 平均PSNR
    """
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)

    Img /= 255.0
    Iclean /= 255.0
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def dct2D(A):
    """
    2D discrete cosine transform (DCT)
    """

    return dct(dct(A, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2D(A):
    """
    inverse 2D discrete cosine transform
    """

    return idct(idct(A, axis=0, norm='ortho'), axis=1, norm='ortho')

def normalize(data):
    return data / 255.

def denormalize(data):
    return np.clip(data * 255., 0, 255).astype(np.uint8)

def show_img(image, title='Image'):
    """顯示單張灰階圖像"""
    image = image.astype(np.float32) / 255.0
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.title(title)
    plt.axis('off')
    plt.show()

def add_gaussian_noise(img, sigma):
    """
    加上高斯雜訊

    :param image: 原圖
    :param sigma: 高斯雜訊標準差
    :return: 高斯雜訊後的圖片
    """
    noise = np.random.normal(0, sigma, img.shape)
    img_noisy = img + noise
    return img_noisy
def get_kaiser_window(blocksize, beta=5):
    """
        產生二維 Kaiser 視窗，用於 patch 加權，避免聚合時產生邊緣偽影。

        Args:
            blocksize (int): 視窗大小，通常與 patch 大小相同（例如 8）
            beta (float): Kaiser 窗的 beta 參數，控制視窗形狀：
                - beta 越小：視窗越平坦（類似矩形窗）
                - beta 越大：視窗越尖銳（集中在中心）
                在 BM3D 中，這會讓 patch 中央的像素在聚合時有更大權重，降低邊緣衝突。

        Returns:
            np.ndarray: shape 為 (blocksize, blocksize) 的 2D 加權矩陣，值域約為 [0, 1]
        """
    k = np.kaiser(blocksize, beta)
    return np.outer(k, k)
def initialization(img, blocksize, kaiser_beta):
    """
    初始化最終圖片、與圖片形狀相同的權重加權、額外加權的kaiser權重

    :param img: 原圖
    :param blocksize: 視窗框大小
    :param kaiser_beta: kaiser beta參數
    :return: 初始化的圖片空間、與空間大小相同的權重圖、kaiser權重
    """
    img_init = np.zeros(img.shape, dtype=float)
    weight_init = np.zeros(img.shape, dtype=float)
    kai = get_kaiser_window(blocksize, kaiser_beta)

    return img_init, weight_init, kai
def SearchWindow(Img, RefPoint, BlockSize, WindowSize):
    """
    Find the search window whose center is reference block in *Img*
    Note that the center of SearchWindow is not always the reference block because of the border
    Return:
        (2 * 2) array of left-top and right-bottom coordinates in search window
    """

    if BlockSize >= WindowSize:
        print('Error: BlockSize is smaller than WindowSize.\n')

        exit()

    Margin = np.zeros((2, 2), dtype=int)
    Margin[0, 0] = max(0, RefPoint[0] + int((BlockSize - WindowSize) / 2))  # left-top x
    Margin[0, 1] = max(0, RefPoint[1] + int((BlockSize - WindowSize) / 2))  # left-top y
    Margin[1, 0] = Margin[0, 0] + WindowSize  # right-bottom x
    Margin[1, 1] = Margin[0, 1] + WindowSize  # right-bottom y

    if Margin[1, 0] >= Img.shape[0]:
        Margin[1, 0] = Img.shape[0] - 1
        Margin[0, 0] = Margin[1, 0] - WindowSize

    if Margin[1, 1] >= Img.shape[1]:
        Margin[1, 1] = Img.shape[1] - 1
        Margin[0, 1] = Margin[1, 1] - WindowSize

    return Margin

def PreDCT(Img, BlockSize):
    """
    Do discrete cosine transform (2D transform) for each block in *Img* to reduce the complexity of
    applying transforms

    Return:
        BlockDCT_all: 4-dimensional array whose first two dimensions correspond to the block's
                      position and last two correspond to the DCT array of the block
    """

    BlockDCT_all = np.zeros((Img.shape[0] - BlockSize, Img.shape[1] - BlockSize, BlockSize, BlockSize), \
                            dtype=float)

    for i in range(BlockDCT_all.shape[0]):
        for j in range(BlockDCT_all.shape[1]):
            Block = Img[i:i + BlockSize, j:j + BlockSize]

            BlockDCT_all[i, j, :, :] = dct2D(Block.astype(np.float64))
            # BlockDCT_all[i, j, :, :] = cv2.dct(Block.astype(np.float64))

    return BlockDCT_all


def Step1_Grouping(noisyImg, RefPoint, BlockDCT_all, BlockSize, ThreDist, MaxMatch, WindowSize):
    """
    Find blocks similar to the reference one in *noisyImg* based on *BlockDCT_all*

    Note that the distance computing is chosen from original paper rather than the analysis one

    Return:
          BlockPos: array of blocks' position (left-top point)
        BlockGroup: 3-dimensional array whose last two dimensions correspond to the DCT array of
                     the block
    """

    # initialization

    WindowLoc = SearchWindow(noisyImg, RefPoint, BlockSize, WindowSize)

    Block_Num_Searched = (WindowSize - BlockSize + 1) ** 2  # number of searched blocks

    BlockPos = np.zeros((Block_Num_Searched, 2), dtype=int)

    BlockGroup = np.zeros((Block_Num_Searched, BlockSize, BlockSize), dtype=float)

    Dist = np.zeros(Block_Num_Searched, dtype=float)

    RefDCT = BlockDCT_all[RefPoint[0], RefPoint[1], :, :]

    match_cnt = 0

    # Block searching and similarity (distance) computing

    for i in range(WindowSize - BlockSize + 1):

        for j in range(WindowSize - BlockSize + 1):

            SearchedDCT = BlockDCT_all[WindowLoc[0, 0] + i, WindowLoc[0, 1] + j, :, :]

            dist = Step1_ComputeDist(RefDCT, SearchedDCT)

            if dist < ThreDist:
                BlockPos[match_cnt, :] = [WindowLoc[0, 0] + i, WindowLoc[0, 1] + j]

                BlockGroup[match_cnt, :, :] = SearchedDCT

                Dist[match_cnt] = dist

                match_cnt += 1

    #    if match_cnt == 1:
    #
    #        print('WARNING: no similar blocks founded for the reference block {} in basic estimate.\n'\
    #              .format(RefPoint))

    if match_cnt <= MaxMatch:

        # less than MaxMatch similar blocks founded, return similar blocks

        BlockPos = BlockPos[:match_cnt, :]

        BlockGroup = BlockGroup[:match_cnt, :, :]

    else:

        # more than MaxMatch similar blocks founded, return MaxMatch similarest blocks

        idx = np.argpartition(Dist[:match_cnt], MaxMatch)  # indices of MaxMatch smallest distances

        BlockPos = BlockPos[idx[:MaxMatch], :]

        BlockGroup = BlockGroup[idx[:MaxMatch], :]

    return BlockPos, BlockGroup
def Step1_ComputeDist(BlockDCT1, BlockDCT2):
    """
    Compute the distance of two DCT arrays *BlockDCT1* and *BlockDCT2*
    """

    if BlockDCT1.shape != BlockDCT1.shape:

        print('ERROR: two DCT Blocks are not at the same shape in step1 computing distance.\n')

        sys.exit()

    elif BlockDCT1.shape[0] != BlockDCT1.shape[1]:

        print('ERROR: DCT Block is not square in step1 computing distance.\n')

        sys.exit()

    BlockSize = BlockDCT1.shape[0]

    if sigma > 40:
        ThreValue = lamb2d * sigma
        BlockDCT1 = np.where(abs(BlockDCT1) < ThreValue, 0, BlockDCT1)
        BlockDCT2 = np.where(abs(BlockDCT2) < ThreValue, 0, BlockDCT2)

    return np.linalg.norm(BlockDCT1 - BlockDCT2) ** 2 / (BlockSize ** 2)
def Step1_3DFiltering(BlockGroup):
    """
    Do collaborative hard-thresholding which includes 3D transform, noise attenuation through
    hard-thresholding and inverse 3D transform

    Return:
        BlockGroup
    """

    ThreValue = lamb3d * sigma
    nonzero_cnt = 0

    # since 2D transform has been done, we do 1D transform, hard-thresholding and inverse 1D
    # transform, the inverse 2D transform is left in aggregation processing

    for i in range(BlockGroup.shape[1]):

        for j in range(BlockGroup.shape[2]):
            ThirdVector = dct(BlockGroup[:, i, j], norm='ortho')  # 1D DCT
            # ThirdVector = cv2.dct(BlockGroup[:, i, j])  # 1D DCT

            ThirdVector[abs(ThirdVector[:]) < ThreValue] = 0.

            nonzero_cnt += np.nonzero(ThirdVector)[0].size

            #BlockGroup[:, i, j] = cv2.idct(ThirdVector).flatten()
            BlockGroup[:, i, j] = list(idct(ThirdVector, norm='ortho'))

    return BlockGroup, nonzero_cnt
def Step1_Aggregation(BlockGroup, BlockPos, basicImg, basicWeight, basicKaiser, nonzero_cnt):
    """
    Compute the basic estimate of the true-image by weighted averaging all of the obtained
    block-wise estimates that are overlapping

    Note that the weight is set accroding to the original paper rather than the BM3D analysis one
    """

    if nonzero_cnt < 1:
        BlockWeight = 1.0 * basicKaiser
    else:
        BlockWeight = (1. / (sigma ** 2 * nonzero_cnt)) * basicKaiser

    for i in range(BlockPos.shape[0]):
        basicImg[BlockPos[i, 0]:BlockPos[i, 0] + BlockGroup.shape[1], \
        BlockPos[i, 1]:BlockPos[i, 1] + BlockGroup.shape[2]] \
            += BlockWeight * idct2D(BlockGroup[i, :, :])
        basicWeight[BlockPos[i, 0]:BlockPos[i, 0] + BlockGroup.shape[1], \
        BlockPos[i, 1]:BlockPos[i, 1] + BlockGroup.shape[2]] += BlockWeight
def BM3D_Step1(img_noisy):
    """
    Give the basic estimate after grouping, collaborative filtering and aggregation

    Return:
        basic estimate basicImg
    """

    # 導入參數
    ThreDist = opt.step1_thre_dist  # threshold distance
    MaxMatch = opt.step1_max_match  # max matched blocks
    blocksize = opt.step1_blocksize
    spdup_factor = opt.step1_spdup  # pixel jump for new reference block
    WindowSize = opt.step1_window_size  # search window size
    kaiser_beta = opt.kaiser_beta
    blockstep = opt.step1_blockstep

    # 初始化

    img_basic, weight_basic, kaiser_basic = initialization(img_noisy, blocksize, kaiser_beta)

    BlockDCT_all = PreDCT(img_noisy, blocksize)

    # block-wise estimate with speed-up factor
    """
        h, w = img_noisy.shape
        #for y in range(0, h - blocksize, blockstep):
        for y in tqdm(range(0, h - blocksize, blockstep), desc='Step1', leave=False):
            for x in range(0, w - blocksize, blockstep):
                RefPoint = [x, y]
    """
    #for i in range(int((img_noisy.shape[0] - blocksize) / spdup_factor) + 2):
    for i in tqdm(range(int((img_noisy.shape[0] - blocksize) / spdup_factor) + 2), desc='Step1'):
        for j in range(int((img_noisy.shape[1] - blocksize) / spdup_factor) + 2):
            RefPoint = [min(spdup_factor * i, img_noisy.shape[0] - blocksize - 1),
                        min(spdup_factor * j, img_noisy.shape[1] - blocksize - 1)]

            BlockPos, BlockGroup = Step1_Grouping(img_noisy, RefPoint, BlockDCT_all, blocksize, \
                                                  ThreDist, MaxMatch, WindowSize)
            BlockGroup, nonzero_cnt = Step1_3DFiltering(BlockGroup)
            Step1_Aggregation(BlockGroup, BlockPos, img_basic, weight_basic, kaiser_basic, nonzero_cnt)

    weight_basic = np.where(weight_basic == 0, 1, weight_basic)
    img_basic[:, :] /= weight_basic[:, :]

    #    basicImg = (np.matrix(basicImg, dtype=int)).astype(np.uint8)

    return img_basic

def Step2_Grouping(basicImg, noisyImg, RefPoint, BlockSize, ThreDist, MaxMatch, WindowSize,
                   BlockDCT_basic, BlockDCT_noisy):
    """
    Similar to Step1_Grouping, find the similar blocks to the reference one from *basicImg*

    Return:
                BlockPos: array of similar blocks' position (left-top point)
        BlockGroup_basic: 3-dimensional array standing for the stacked blocks similar to the
                          reference one from *basicImg* after 2D DCT
        BlockGroup_noisy: the stacked blocks from *noisyImg* corresponding to BlockGroup_basic
    """

    # initialization (same as Step1)

    WindowLoc = SearchWindow(basicImg, RefPoint, BlockSize, WindowSize)

    Block_Num_Searched = (WindowSize - BlockSize + 1) ** 2

    BlockPos = np.zeros((Block_Num_Searched, 2), dtype=int)

    BlockGroup_basic = np.zeros((Block_Num_Searched, BlockSize, BlockSize), dtype=float)

    BlockGroup_noisy = np.zeros((Block_Num_Searched, BlockSize, BlockSize), dtype=float)

    Dist = np.zeros(Block_Num_Searched, dtype=float)

    match_cnt = 0

    # Block searching and similarity (distance) computing
    # Note the distance computing method is different from that of Step1

    for i in range(WindowSize - BlockSize + 1):

        for j in range(WindowSize - BlockSize + 1):

            SearchedPoint = [WindowLoc[0, 0] + i, WindowLoc[0, 1] + j]

            dist = Step2_ComputeDist(basicImg, RefPoint, SearchedPoint, BlockSize)

            if dist < ThreDist:
                BlockPos[match_cnt, :] = SearchedPoint

                Dist[match_cnt] = dist

                match_cnt += 1

    #    if match_cnt == 1:
    #
    #        print('WARNING: no similar blocks founded for the reference block {} in final estimate.\n'\
    #              .format(RefPoint))

    if match_cnt <= MaxMatch:

        # less than MaxMatch similar blocks founded, return similar blocks

        BlockPos = BlockPos[:match_cnt, :]

    else:

        # more than MaxMatch similar blocks founded, return MaxMatch similarest blocks

        idx = np.argpartition(Dist[:match_cnt], MaxMatch)  # indices of MaxMatch smallest distances

        BlockPos = BlockPos[idx[:MaxMatch], :]

    for i in range(BlockPos.shape[0]):
        SimilarPoint = BlockPos[i, :]

        BlockGroup_basic[i, :, :] = BlockDCT_basic[SimilarPoint[0], SimilarPoint[1], :, :]

        BlockGroup_noisy[i, :, :] = BlockDCT_noisy[SimilarPoint[0], SimilarPoint[1], :, :]

    BlockGroup_basic = BlockGroup_basic[:BlockPos.shape[0], :, :]

    BlockGroup_noisy = BlockGroup_noisy[:BlockPos.shape[0], :, :]

    return BlockPos, BlockGroup_basic, BlockGroup_noisy


def Step2_ComputeDist(img, Point1, Point2, BlockSize):
    """
    Compute distance between blocks whose left-top margins' coordinates are *Point1* and *Point2*
    """

    Block1 = (img[Point1[0]:Point1[0] + BlockSize, Point1[1]:Point1[1] + BlockSize]).astype(np.float64)

    Block2 = (img[Point2[0]:Point2[0] + BlockSize, Point2[1]:Point2[1] + BlockSize]).astype(np.float64)

    return np.linalg.norm(Block1 - Block2) ** 2 / (BlockSize ** 2)
def Step2_3DFiltering(BlockGroup_basic, BlockGroup_noisy):
    """
    Do collaborative Wiener filtering and here we choose 2D DCT + 1D DCT as the 3D transform which
    is the same with the 3D transform in hard-thresholding filtering

    Note that the Wiener weight is set accroding to the BM3D analysis paper rather than the original
    one

    Return:
       BlockGroup_noisy & WienerWeight
    """

    Weight = 0

    coef = 1.0 / BlockGroup_noisy.shape[0]

    for i in range(BlockGroup_noisy.shape[1]):

        for j in range(BlockGroup_noisy.shape[2]):
            Vec_basic = dct(BlockGroup_basic[:, i, j], norm='ortho')

            Vec_noisy = dct(BlockGroup_noisy[:, i, j], norm='ortho')

            Vec_value = Vec_basic ** 2 * coef

            Vec_value /= (Vec_value + sigma ** 2)  # pixel weight

            Vec_noisy *= Vec_value

            Weight += np.sum(Vec_value)
            #            for k in range(BlockGroup_noisy.shape[0]):
            #
            #                Value = Vec_basic[k]**2 * coef
            #
            #                Value /= (Value + sigma**2) # pixel weight
            #
            #                Vec_noisy[k] = Vec_noisy[k] * Value
            #
            #                Weight += Value

            BlockGroup_noisy[:, i, j] = list(idct(Vec_noisy, norm='ortho'))


    if Weight > 0:

        WienerWeight = 1. / (sigma ** 2 * Weight)

    else:

        WienerWeight = 1.0

    return BlockGroup_noisy, WienerWeight
def Step2_Aggregation(BlockGroup_noisy, WienerWeight, BlockPos, finalImg, finalWeight, finalKaiser):
    """
    Compute the final estimate of the true-image by aggregating all of the obtained local estimates
    using a weighted average
    """

    BlockWeight = WienerWeight * finalKaiser

    for i in range(BlockPos.shape[0]):
        finalImg[BlockPos[i, 0]:BlockPos[i, 0] + BlockGroup_noisy.shape[1], \
        BlockPos[i, 1]:BlockPos[i, 1] + BlockGroup_noisy.shape[2]] \
            += BlockWeight * idct2D(BlockGroup_noisy[i, :, :])
        finalWeight[BlockPos[i, 0]:BlockPos[i, 0] + BlockGroup_noisy.shape[1], \
        BlockPos[i, 1]:BlockPos[i, 1] + BlockGroup_noisy.shape[2]] += BlockWeight
def BM3D_Step2(basicImg, noisyImg):
    """
    Give the final estimate after grouping, Wiener filtering and aggregation

    Return:
        final estimate finalImg
    """

    # parameters setting

    ThreDist = opt.step2_thre_dist
    MaxMatch = opt.step2_max_match
    BlockSize = opt.step2_blocksize
    spdup_factor = opt.step2_spdup
    WindowSize = opt.step2_window_size


    finalImg, finalWeight, finalKaiser = initialization(basicImg, BlockSize, Kaiser_Window_beta)
    BlockDCT_noisy = PreDCT(noisyImg, BlockSize)
    BlockDCT_basic = PreDCT(basicImg, BlockSize)

    # block-wise estimate with speed-up factor

    # for i in range(int((basicImg.shape[0] - BlockSize) / spdup_factor) + 2):
    for i in tqdm(range(int((img_noisy.shape[0] - BlockSize) / spdup_factor) + 2), desc='Step2'):
        for j in range(int((basicImg.shape[1] - BlockSize) / spdup_factor) + 2):
            RefPoint = [min(spdup_factor * i, basicImg.shape[0] - BlockSize - 1), \
                        min(spdup_factor * j, basicImg.shape[1] - BlockSize - 1)]

            BlockPos, BlockGroup_basic, BlockGroup_noisy = Step2_Grouping(basicImg, noisyImg, \
                                                                          RefPoint, BlockSize, \
                                                                          ThreDist, MaxMatch, \
                                                                          WindowSize, \
                                                                          BlockDCT_basic, \
                                                                          BlockDCT_noisy)

            BlockGroup_noisy, WienerWeight = Step2_3DFiltering(BlockGroup_basic, BlockGroup_noisy)

            Step2_Aggregation(BlockGroup_noisy, WienerWeight, BlockPos, finalImg, finalWeight, \
                              finalKaiser)

    finalWeight = np.where(finalWeight == 0, 1, finalWeight)

    finalImg[:, :] /= finalWeight[:, :]

    #    finalImg = (np.matrix(finalImg, dtype=int)).astype(np.uint8)

    return finalImg


if __name__ == '__main__':

    # 參數解析
    # ============================================================================================
    parser = argparse.ArgumentParser(description="BM3D")
    parser.add_argument("--folder_path", type=str, default='data/Set1', help='Path to test image')
    parser.add_argument("--sigma", type=float, default=25, help='高斯噪音的標準差')
    parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')

    # Step 1 參數
    parser.add_argument("--step1_thre_dist", type=float, default=2500, help='Step1 threshold distance')
    parser.add_argument("--step1_max_match", type=int, default=16, help='Step1 max matched blocks')
    parser.add_argument("--step1_blocksize", type=int, default=8, help='Step1 block size')
    parser.add_argument("--step1_blockstep", type=int, default=3, help='Step1 block step')
    parser.add_argument("--step1_spdup", type=int, default=3, help='Step1 speed-up factor')
    parser.add_argument("--step1_window_size", type=int, default=39, help='Step1 search window size')

    # Step 2 參數
    parser.add_argument("--step2_thre_dist", type=float, default=400, help='Step2 threshold distance')
    parser.add_argument("--step2_max_match", type=int, default=32, help='Step2 max matched blocks')
    parser.add_argument("--step2_blocksize", type=int, default=8, help='Step2 block size')
    parser.add_argument("--step2_blockstep", type=int, default=3, help='Step2 block step')
    parser.add_argument("--step2_spdup", type=int, default=3, help='Step2 speed-up factor')
    parser.add_argument("--step2_window_size", type=int, default=39, help='Step2 search window size')

    # 通用參數
    parser.add_argument("--kaiser_beta", type=float, default=2.0, help='Kaiser window beta')
    opt = parser.parse_args()
    sigma = opt.sigma  # 高斯雜訊標準差
    Kaiser_Window_beta = opt.kaiser_beta
    cv2.setUseOptimized(True)

        # 參數說明
        # blocksize (int): 每個 patch 的邊長（正方形）。
        # max_match (int): 3D疊加的stack層數上限
        # blockstep (int): 搜尋視窗內滑動的步長。
        # spdup (int): 搜尋參考座標點時的像素間格，越大間格就越大、越省時間，概念與blockstep相同
        # searchsize (int): 搜尋視窗半徑，會在 (y, x) 周圍 2*searchsize 範圍內搜尋。
        # diffT (float): 計算兩區塊差異的門檻，控制進入 group 的相似性
        # coefT (float): DCT 係數的閾值，小於此值的會被設為 0

    lamb2d = 2.0
    lamb3d = 2.7

    psnr_basic_total = 0
    psnr_final_total = 0

    # 讀入影像與預處理
    # ============================================================================================
    print('Loading data ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()
        # 讀取所有影像
    output_dir = f"demo_BM3D_hand"
    os.makedirs(output_dir, exist_ok=True)
        # 創建輸出資料夾

    # 遍歷處理影像
    # ============================================================================================
    for idx, f in enumerate(files_source, start=1):

        print('idx,',idx)
        #for image_path in tqdm(image_paths, desc='Processing Images'):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        #img = cv2.imread(image_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # show_img(img, title='Original Image')
        # print("讀取影像:", image_path)
        # print("影像尺寸:", img.shape)
        img_noisy = add_gaussian_noise(img, sigma)      # 加入高斯雜訊
        # show_img(img_noisy, title='Noisy Image')

        start_time = time.time()                        # SSIM用計時
        img_basic = BM3D_Step1(img_noisy)               # 第一階段處理
        step1_time = time.time()                        # 第一階段計時
        print('The running time of basic estimate is', step1_time - start_time, 'seconds.\n')

        img_final = BM3D_Step2(img_basic, img_noisy)    # 第二階段處理
        step2_time = time.time()                        # 第二階段計時
        print('The running time of final estimate is', step2_time - step1_time, 'seconds.\n')

        # PSNR計算
        # ============================================================================================
                                                        # 計算第一階段平均PSNR
        psnr_basic = batch_PSNR(torch.Tensor(img).unsqueeze(0).unsqueeze(0),
                          torch.Tensor(img_basic).unsqueeze(0).unsqueeze(0), 1.)
        print('psnr basic:',psnr_basic)
        psnr_basic_total += psnr_basic
                                                        # 計算第二階段平均PSNR
        psnr_final = batch_PSNR(torch.Tensor(img).unsqueeze(0).unsqueeze(0),
                                torch.Tensor(img_final).unsqueeze(0).unsqueeze(0), 1.)
        print('psnr final:',psnr_final)
        psnr_final_total += psnr_final

        # 圖片繪製
        # ============================================================================================
        stacked = np.hstack((img, img_noisy, img_basic, img_final))
        titled = add_titles(stacked, ['Original', 'Noisy', 'Basic', 'Final'])
        output_path = os.path.join(output_dir, f"{idx:04d}_compare.png")
        cv2.imwrite(output_path, titled)

    psnr_final_avg = psnr_final_total / len(files_source)
    print(f"\nfinal PSNR on test data: {psnr_final_avg:.6f}")

    psnr_basic_avg = psnr_final_total / len(files_source)
    print(f"\nbasic PSNR on test data: {psnr_basic_avg:.6f}")