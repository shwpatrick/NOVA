import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

def add_gaussian_noise(image, sigma):
    """
    加上高斯雜訊
    clip 可以防止雜訊溢出

    :param image:
    :param sigma:
    :return:
    """
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 1)
def prepare_image(image, blocksize, blockstep, searchsize):
    """
    對雜訊影像進行邊界處理與尺寸對齊。

    目的：
      - 讓影像的寬高可以剛好被 block size 與步長整除
      - 避免後續 DCT 時 patch 超出邊界

    Args:
        image (np.ndarray): 輸入影像，通常為灰階且已正規化至 [0, 1]
        blocksize (int): 區塊大小，例如 4 表示每個 patch 是 4x4
        blockstep (int): 區塊掃描的步長，例如 2 表示每次移動 2 個像素
        searchsize (int): 搜尋視窗半徑，例如 16 表示以每個 patch 為中心向外搜尋 16 像素內的相似 patch

    Returns:
        np.ndarray: 經過尺寸修正後的新影像，適用於 BM3D 運算。
    """
    h, w = image.shape
    newh = math.floor((h - blocksize) / blockstep) * blockstep - searchsize
    neww = math.floor((w - blocksize) / blockstep) * blockstep - searchsize
    newh = math.floor((h - newh - blocksize) / blockstep) * blockstep + newh
    neww = math.floor((w - neww - blocksize) / blockstep) * blockstep + neww
    return image[0:newh, 0:neww]
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
def show_img(image, title='Image'):
    """顯示單張灰階圖像"""
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.axis('off')
    plt.show()
def show_DCT_patch_after_IDCT(stack):
    for i in range(stack.shape[2]):
        restored = cv2.idct(stack[:, :, i])
        # show or save
        plt.imshow(restored, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Restored patch #{i}")
        plt.axis('off')
        plt.show()
def process_bar(percent, start_str='', end_str='', total_length=0):
    """外層迴圈中，用來動態顯示目前處理進度的進度條"""
    bar = ''.join(["\033[31m%s\033[0m" % '   '] * int(percent * total_length)) + ''
    bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent * 100) + end_str
    print(bar, end='', flush=True)
def find_similar_blocks_step1(imnoise, y, x, blocksize, searchsize, blockstep, blockmaxnum, diffT):
    """
    在指定位置 (y, x) 的參考 patch 周圍，搜尋相似的 patch 並組成 3D 區塊。

    Args:
        imnoise (np.ndarray): 輸入的帶雜訊影像。
        y (int): 參考 patch 的左上角 y 座標。
        x (int): 參考 patch 的左上角 x 座標。
        blocksize (int): 每個 patch 的邊長（正方形）。
        searchsize (int): 搜尋視窗半徑，會在 (y, x) 周圍 2*searchsize 範圍內搜尋。
        blockstep (int): 搜尋視窗內滑動的步長。
        blockmaxnum (int): 最多保留的相似 patch 數量（含自己）。
        diffT (float): 相似度門檻，小於此距離的 patch 才會被視為相似。

    Returns:
        tuple:
            - similarBoxArray (np.ndarray): shape = (blocksize, blocksize, blockmaxnum) 的 3D 區塊。
            - hasboxnum (int): 實際找到的相似區塊數量。
            - coords: 每個相似 patch 左上角的 (y, x) 座標
    """

    # 初始化
    newh, neww = imnoise.shape
    similarBoxArray = np.zeros((blocksize, blocksize, blockmaxnum))
        # 建立一個三維陣列 similarBoxArray 來存放最多 blockmaxnum 個相似區塊（patch）
    diffArray = np.zeros(blockmaxnum)
        # diffArray 用來記錄每個區塊與參考 patch 的距離，用來排序

    similarBoxArray[:, :, 0] = imnoise[y:y + blocksize, x:x + blocksize]
        # 第一個位置保留自己（reference block）
    diffArray[0] = 0
    hasboxnum = 1
        # hasboxnum 記錄目前已放入幾個有效 patch

    systart = max(0, y - searchsize)
    syend = min(newh - blocksize, y + searchsize - 1)
    sxstart = max(0, x - searchsize)
    sxend = min(neww - blocksize, x + searchsize - 1)
        # 避免搜尋視窗超出圖像邊界
        # searchsize 是半徑，實際搜尋為 (2 * searchsize + 1)^2 個候選 patch

    #  掃描搜尋區塊（block matching）
    for sy in range(systart, syend, blockstep):
        for sx in range(sxstart, sxend, blockstep):
            if sy == y and sx == x:
                continue
            # 計算相似度（距離）
            # 這裡使用的是 L1 距離
            # diff = np.sum(np.abs(patch_P - patch_Q))
            # Patch P：參考區塊（reference patch）
            # Patch Q：候選區塊（candidate patch）
            diff = np.sum(np.abs(
                imnoise[y: y + blocksize, x: x + blocksize] - imnoise[sy: sy + blocksize, sx: sx + blocksize]))
            if diff > diffT:
                continue

            # 維護一個最多 blockmaxnum 個相似 patch 的 3D 群組
            changeid = 0                    # 預設不新增任何 patch
            if hasboxnum < blockmaxnum: # 目前還沒收滿 blockmaxnum 個 patch
                changeid = hasboxnum        # 直接把當前這個相似 patch（Q）放進第 hasboxnum 個位置
                hasboxnum += 1
            else:                                        # 已經滿了
                for difid in range(1, blockmaxnum):  # 考慮「替換掉現有比較差的 patch」
                    if diff < diffArray[difid]:
                        changeid = difid

            # 當changeid != 0 就代表，新的候選窗符合條件且還沒滿，或者滿了但比其中最大相似性的還小
            # 就可以依照changeid 位置把它替換掉
            if changeid != 0:
                similarBoxArray[:, :, changeid] = imnoise[sy: sy + blocksize, sx: sx + blocksize]
                diffArray[changeid] = diff

    return similarBoxArray, hasboxnum
def find_similar_blocks_step2(imnoise, imbasic, y, x, blocksize, searchsize, blockstep, blockmaxnum, diffT, block_DCT_noisy, block_DCT_basic):
    """
    Step2
    「只用 basic image 來進行 block matching」
    然後 依據找出的 patch 位置，
    去 basic image 與 noisy image 各自抓對應 patch

    :param imnoise:
    :param basic_estimate:
    :param y:
    :param x:
    :param blocksize:
    :param searchsize:
    :param blockstep:
    :param blockmaxnum:
    :param diffT:
    :return:
    """
    newh, neww = imnoise.shape
    similarBoxArray_noisy = np.zeros((blocksize, blocksize, blockmaxnum))
    similarBoxArray_basic = np.zeros((blocksize, blocksize, blockmaxnum))
        # 建立一個三維陣列 similarBoxArray 來存放最多 blockmaxnum 個相似區塊（patch）

    coords = np.zeros((blockmaxnum, 2), dtype=int)
        # 用來保存座標參考
    diffArray = np.zeros(blockmaxnum)
        # diffArray 用來記錄每個區塊與參考 patch 的距離，用來排序
    similarBoxArray_basic[:, :, 0] = imbasic[y:y + blocksize, x:x + blocksize]
        # 第一個位置保留自己（reference block）
    coords[0, :] = (y, x)
    diffArray[0] = 0

    hasboxnum = 1
        # hasboxnum 記錄目前已放入幾個有效 patch

    systart = max(0, y - searchsize)
    syend = min(newh - blocksize, y + searchsize - 1)
    sxstart = max(0, x - searchsize)
    sxend = min(neww - blocksize, x + searchsize - 1)
    # 避免搜尋視窗超出圖像邊界
    # searchsize 是半徑，實際搜尋為 (2 * searchsize + 1)^2 個候選 patch

    #  掃描搜尋區塊（block matching）
    for sy in range(systart, syend, blockstep):
        for sx in range(sxstart, sxend, blockstep):
            if sy == y and sx == x:
                continue
            # 計算相似度（距離）
            # 這裡使用的是 L1 距離
            # diff = np.sum(np.abs(patch_P - patch_Q))
            # Patch P：參考區塊（reference patch）
            # Patch Q：候選區塊（candidate patch）
            diff = np.sum(np.abs(
                imbasic[y: y + blocksize, x: x + blocksize] - imbasic[sy: sy + blocksize, sx: sx + blocksize]))
            if diff > diffT:
                continue

            # 維護一個最多 blockmaxnum 個相似 patch 的 3D 群組
            changeid = 0  # 預設不新增任何 patch
            if hasboxnum < blockmaxnum - 1:  # 目前還沒收滿 blockmaxnum - 1 個 patch
                changeid = hasboxnum  # 直接把當前這個相似 patch（Q）放進第 hasboxnum 個位置
                hasboxnum += 1
            else:  # 已經滿了
                for difid in range(1, blockmaxnum - 1):  # 考慮「替換掉現有比較差的 patch」
                    if diff < diffArray[difid]:
                        changeid = difid

            # 當changeid != 0 就代表，新的候選窗符合條件且還沒滿，或者滿了但比其中最大相似性的還小
            # 就可以依照changeid 位置把它替換掉
            if changeid != 0:
                similarBoxArray_basic[:, :, changeid] = imnoise[sy: sy + blocksize, sx: sx + blocksize]
                diffArray[changeid] = diff
                coords[changeid, :] = (sy, sx)

    # 統計
    if hasboxnum == 1:
        print('WARNING: no similar blocks founded for the reference block')
    #        print('WARNING: no similar blocks founded for the reference block {} in final estimate.\n'\
    #              .format(RefPoint))



    for i in range(coords.shape[0]):
        point = coords[i, :]
        similarBoxArray_basic[:, :, i] = block_DCT_basic[point[0], point[1], :, :]
        similarBoxArray_noisy[:, :, i] = block_DCT_noisy[point[0], point[1], :, :]

    similarBoxArray_basic = similarBoxArray_basic[:, :, :coords.shape[0]]
    similarBoxArray_noisy = similarBoxArray_noisy[:, :, :coords.shape[0]]

    return coords, similarBoxArray_basic, similarBoxArray_noisy
def pre_DCT(img, blocksize):
    block_DCT = np.zeros((img.shape[0] - blocksize, img.shape[1] - blocksize, blocksize, blocksize), dtype = float)
    for i in range(block_DCT.shape[0]):
        for j in range(block_DCT.shape[1]):
            block = img[i:i + blocksize, j:j + blocksize].astype(np.float64)
            block_DCT[i,j,:,:] = cv2.dct(block)
    return block_DCT
def bm3d_step1(imnoise, opt):
    """
    執行 BM3D 第一階段：以硬閾值協同濾波對影像進行降噪。

    Args:
        imnoise (np.ndarray): 帶有高斯雜訊的灰階圖像，值域應在 [0, 1]
        sigma (float): 雜訊標準差，用來控制濾波強度
        blocksize (int): 每個 patch 的大小，BM3D 論文預設為 8
        blockstep (int): 區塊掃描步長，BM3D 論文預設為 3，表示區塊會有重疊
        searchsize (int): 以參考 patch 為中心，搜尋相似區塊的視窗半徑，預設為 16
        blockmaxnum (int): 每個參考 patch 最多保留的相似區塊數量
        diffT (float): 計算兩區塊差異的門檻，控制進入 group 的相似性
        coefT (float): DCT 係數的閾值，小於此值的會被設為 0

    Returns:
        np.ndarray: 降噪後的影像，與輸入尺寸相同
    """

    # 使用參數
    # ==================================================================
    sigma = opt.sigma
    blocksize = opt.step1_blocksize
    blockstep = opt.step1_blockstep
    searchsize = opt.step1_searchsize
    blockmaxnum = opt.step1_blockmaxnum
    diffT = opt.step1_diffT
    coefT = opt.step1_coefT

    # 執行第一階段 BM3D 的 hard-threshold 去噪

    # 初始化
    # ==================================================================
    size = imnoise.shape
    h, w = size                        # 取出影像寬高
    imnum = np.zeros(size)             # 影像緩衝區：imnum（分子），最後要做 weighted average
    imden = np.zeros(size)             # 影像緩衝區：imden（分母），最後要做 weighted average
    kai = get_kaiser_window(blocksize) #  Kaiser window 作為區塊內的權重模板
        # Kaiser window 是一種可調參數的視窗函數，常用於信號處理中做加權濾波或邊界抑制。在 BM3D 中，它用於：
            # 對每個影像區塊（patch）內的像素進行空間加權
            # 讓 patch 中央的像素具有較高的權重，邊緣的權重較小
            # 避免在重疊 patch 聚合時產生邊界偽影（artifact）


    # 顯示進度條
    for y in tqdm(range(0, h - blocksize, blockstep), desc='Step1', leave=False):
    # for y in range(0, h - blocksize, blockstep):
        # 顯示進度條
        # print("h y type:", type(newh), type(y))
        # process_bar(y / (h - blocksize), start_str='進度', end_str="100", total_length=15)

        for x in range(0, w - blocksize, blockstep):

            # Step1_Grouping（分群匹配）、相似區塊分組（Block Matching)
            # ============================================================================================
            similarBoxArray, hasboxnum = find_similar_blocks_step1(imnoise, y, x, blocksize, searchsize, blockstep, blockmaxnum, diffT)

            # Step1_3DFiltering（3D 濾波）
            # ============================================================================================
            # 協同濾波步驟（Collaborative Filtering）
            """
            3D sparsifying transform 的設計：
                2D DCT：針對每個 patch 單獨做頻域轉換（橫 + 直方向）
                1D DCT：針對相同位置的像素沿「z 軸（patch stack）」做轉換（相當於橫跨 patch）
            
            這樣就能充分利用影像中：
                每個 patch 的區域性結構（透過 2D DCT）
                patch 之間的重複性（透過 1D DCT）
            """
            # 對每個 patch 做  2D DCT（離散餘弦轉換），讓它進入頻率域
            # 0 是參考 patch P，在第二階段會用，這裡不處理
            for i in range(1, hasboxnum): # hasboxnum是3D群組的z軸數量
                similarBoxArray[:, :, i] = cv2.dct(similarBoxArray[:, :, i])

            # 進行 1D DCT 與硬閾值處理
            notzeronum = 0
            for y1d in range(blocksize):                               # 針對每一個 pixel 位置 (y1d, x1d)
                for x1d in range(blocksize):
                    temp3ddct = cv2.dct(similarBoxArray[y1d, x1d, :])  # 對這條向量做 1D DCT → 得到該位置在 patch 群組中的頻率資訊
                    zeroidx = np.abs(temp3ddct) < coefT
                        # 找出 temp3ddct 中，絕對值小於閾值 coefT 的元素，
                        # 這些值被視為雜訊或微弱訊號，應該清除為 0
                    temp3ddct[zeroidx] = 0
                    notzeronum += np.count_nonzero(temp3ddct)
                        # 計算並累加目前這個像素座標 (y1d, x1d) 上，
                        # 所有 patch 經過 1D DCT + 硬閾值處理後的 非零係數數量
                        # 用來評估這個 3D 區塊的「結構複雜度」或「稀疏性」
                        # 非零係數多 → 結構複雜（變化大） → 權重應小
                        # 非零係數少 → 結構簡單（一致性高）→ 權重可以大
                        #
                        # 我的解釋：
                        # 在頻率域中，低幅度的係數很可能是雜訊，經過硬閾值處理會被清成 0。
                        # 若壓縮後留下很多非零係數 → 表示這個 3D patch group 結構豐富、變化大。
                        # 但這樣的 patch group 在估計時可能不夠「一致」，推論出的 pixel 值會比較不穩定、不可靠。
                        # 因此在加權（aggregation）時，應該給這類 group 較小的權重，降低它對最終影像的影響力。
                    similarBoxArray[y1d, x1d, :] = cv2.idct(temp3ddct)[:, 0]
                        # 將剛剛處理過的 1D 頻率資料 temp3ddct 還原回空間域
            notzeronum = max(notzeronum, 1)
                        # 避免後續權重計算中出現除以 0 的錯誤。

            # Step1_Aggregation（加權聚合）、聚合回原圖（加權）
            # ============================================================================================
            # 將經過 3D 濾波後的每個 patch 加回到原圖中正確位置
            weight = kai / ((sigma ** 2) * notzeronum)
                # kai：Kaiser window，對 patch 中心像素權重較大，邊緣小
                # sigma²：越大的雜訊 → 濾波越強 → 權重要更保守
                # notzeronum：保留的 DCT 頻率數量越多 → 結構越複雜 → 可信度越低 → 權重要小
            for i in range(1, hasboxnum):
                imidct = cv2.idct(similarBoxArray[:, :, i])                  # 還原回空間域 patch
                imnum[y: y + blocksize, x: x + blocksize] += imidct * weight
                imden[y: y + blocksize, x: x + blocksize] += weight
                # 在原本空的圖上面把加權疊上去
                # 加到影像的分子與分母上
                # 分子 imnum：累積所有加權後的 patch 預測值
                # 分母 imden：累積權重

    return imnum / (imden + 1e-8)
def bm3d_step2(imnoise, imbasic, opt):
    """
    BM3D 第二階段：使用 Wiener Filtering 進一步降噪。

    第二階段使用不同的參考圖來找相似patch 還是做block matching
    第一階段使用的是 noisy image 本身（有很多雜訊），所以patch 很容易失準
    第二階段使用的是 basic estimate（第一階段輸出），組成的 3D group 質量更好

    Args:
        imnoise (np.ndarray): 原始加雜訊影像。
        basic_estimate (np.ndarray): 第一階段輸出的基本估計。
        sigma (float): 雜訊標準差。
        blocksize, blockstep, searchsize, blockmaxnum, coefT: 同第一階段

    Returns:
        np.ndarray: 第二階段降噪結果。
    """

    # 使用參數
    # ==================================================================
    sigma = opt.sigma
    blocksize = opt.step2_blocksize
    blockstep = opt.step2_blockstep
    searchsize = opt.step2_searchsize
    blockmaxnum = opt.step2_blockmaxnum
    diffT = opt.step1_diffT
    coefT = opt.step1_coefT

    # 初始化
    # ==================================================================
    size = imnoise.shape
    h, w = size                         # 取出影像寬高
    imnum = np.zeros(size)              # 影像緩衝區：imnum（分子），最後要做 weighted average
    imden = np.zeros(size)              # 影像緩衝區：imden（分母），最後要做 weighted average
    kai = get_kaiser_window(blocksize)  # Kaiser window 作為區塊內的權重模板
        # Kaiser window 是一種可調參數的視窗函數，常用於信號處理中做加權濾波或邊界抑制。在 BM3D 中，它用於：
        # 對每個影像區塊（patch）內的像素進行空間加權
        # 讓 patch 中央的像素具有較高的權重，邊緣的權重較小
        # 避免在重疊 patch 聚合時產生邊界偽影（artifact）

    block_DCT_noisy = pre_DCT(imnoise, blocksize)
    block_DCT_basic = pre_DCT(imbasic, blocksize)
        # 把每一個可能的 blocksize × blocksize 區塊先 DCT 好
        # 存進一個 shape = (h, w, B, B) 的 4D tensor 中
        # 後面找 patch 時直接取用即可（類似 cache）

    imfinal = np.zeros(imbasic.shape, dtype=float)
    weight_final = np.zeros(imbasic.shape, dtype=float)

    # 顯示進度條
    for y in tqdm(range(0, h - blocksize, blockstep), desc='Step1', leave=False):
    #for y in range(0, h - blocksize, blockstep):
        # process_bar(y / (h - blocksize), start_str='進度', end_str="100", total_length=15)
        for x in range(0, w - blocksize, blockstep):

            # Step2_Grouping
            # ==================================================================
            # similarBoxArray, hasboxnum = find_similar_blocks_step(imnoise, y, x, blocksize, searchsize, blockstep, blockmaxnum, diffT)
            coords, similarBoxArray_basic, similarBoxArray_noisy = find_similar_blocks_step2(imnoise, imbasic, y, x,
                                      blocksize, searchsize, blockstep, blockmaxnum, diffT,
                                      block_DCT_noisy, block_DCT_basic)

            # show_DCT_patch_after_IDCT(similarBoxArray_basic)
            # show_DCT_patch_after_IDCT(similarBoxArray_noisy)

            # Step2_3DFiltering
            # ==================================================================
            # 這裡的針對的目標是x,y座標所對應的patch
            # 在已經組成的 3D patch 群組中，對沿 z 軸（patch 堆疊）進行 1D DCT + Wiener shrinkage，再進行還原（IDCT）
            # ((blocksize, blocksize, blockmaxnum))
            weight = 0 # 累積整個 3D block 的 Wiener shrink factor 總量
            coef = 1.0 / similarBoxArray_noisy.shape[2]
                # 正規化係數（patch 數量的倒數），用於估計 signal energy（類似平均）
                # 3Dstack層數分之一

            # 遍歷 patch 中的每一個 pixel 位置 (i, j)
            # 做1D DCT 形成一條向量
            for i in range(similarBoxArray_noisy.shape[0]):
                for j in range(similarBoxArray_noisy.shape[1]):
                    vec_basic = cv2.dct(similarBoxArray_basic[i, j, :])
                    vec_noisy = cv2.dct(similarBoxArray_noisy[i, j, :])
                    vec_value = vec_basic ** 2 * coef     # 根據 vec_basic 的平方來估計 signal power
                    vec_value /= (vec_value + sigma ** 2) # 計算 Wiener shrinkage 系數（gain）
                    vec_noisy *= vec_value                # 濾波 noisy 的頻率值
                                                          # 根據 Wiener shrinkage 係數縮放 noisy 頻率分量（即濾波）
                    weight += np.sum(vec_value)           # 更新整體權重
                                                          # 將所有 shrinkage 系數加總，作為該 block 聚合時的總能量指標
                                                          # 這會被用來算 Kaiser weighted aggregation 的權重
                                                          # （越「乾淨」的 patch 給越高權重）

                    similarBoxArray_noisy[i, j, :] = cv2.idct(vec_noisy)[:, 0]
                                                          # idct後的結果 shape 是 (N, 1)
                                                          # [:, 0] 是為了把它還原成 shape (N,) 的 1D 向量
                                                          # 好塞回 similarBoxArray[i, j, :]

            #  Wiener 聚合權重（WienerWeight） 的計算邏輯
            if weight > 0:
                weight_wiener = 1. / (sigma ** 2 * weight)
            else:
                weight_wiener = 1.0

            # Step2_Aggregation
            # ==================================================================
            weight_block = weight_wiener * kai

            for i in range(coords.shape[0]):
                point = coords[i, :]
                imfinal[point[0]:point[0]+similarBoxArray_noisy.shape[0], point[1]:point[1]+similarBoxArray_noisy.shape[1]] \
                    += weight_block * cv2.idct(similarBoxArray_noisy[:, :, i])

                weight_final[point[0]:point[0]+similarBoxArray_noisy.shape[0], point[1]:point[1]+similarBoxArray_noisy.shape[1]] \
                    += weight_block

                # finalImg[BlockPos[i, 0]:BlockPos[i, 0] + BlockGroup_noisy.shape[1], \
                # BlockPos[i, 1]:BlockPos[i, 1] + BlockGroup_noisy.shape[2]] \
                #     += BlockWeight * idct2D(BlockGroup_noisy[i, :, :])

                # finalWeight[BlockPos[i, 0]:BlockPos[i, 0] + BlockGroup_noisy.shape[1], \
                # BlockPos[i, 1]:BlockPos[i, 1] + BlockGroup_noisy.shape[2]] += BlockWeight

            weight_final = np.where(weight_final == 0, 1, weight_final)
                # finalWeight 中有任何像素值為 0，就把它改成 1，其他保持不變
                # 如果有某個 pixel 完全沒有被任何 patch 覆蓋，那麼它在 finalWeight 中就會是 0

            imfinal[:, :] /= weight_final[:, :]
                # 加權平均（weighted average）

    return imfinal


if __name__ == "__main__":

    # 參數解析
    # ============================================================================================
    parser = argparse.ArgumentParser(description="BM3D")
    parser.add_argument("--image_path", type=str, default='data/Set12/01.png', help='Path to test image')
    parser.add_argument("--sigma", type=float, default=0.03, help='高斯噪音的標準差，因為已經正則化過，0.03算小的')
    parser.add_argument("--step1_diffT", type=float, default=100, help='Step1 diffT')
    parser.add_argument("--step1_coefT", type=float, default=0.0005, help='Step1 coefT')
    parser.add_argument("--step1_blocksize", type=int, default=8, help='Step1 Patch size')
    parser.add_argument("--step1_blockstep", type=int, default=3, help='Step1 Block sliding step')
    parser.add_argument("--step1_blockmaxnum", type=int, default=16, help='Step1 Block Max Match Numbers')
    parser.add_argument("--step1_searchsize", type=int, default=16, help='Step1 Search window radius')
    parser.add_argument("--step2_blocksize", type=int, default=8, help='Step2 Patch size')
    parser.add_argument("--step2_blockstep", type=int, default=3, help='Step2 Block sliding step')
    parser.add_argument("--step2_searchsize", type=int, default=16, help='Step2 Search window radius')
    parser.add_argument("--step2_blockmaxnum", type=int, default=16, help='Step2 Block Max Match Numbers')
    parser.add_argument("--blockmaxnum", type=int, default=16, help='Maximum number of similar patches to group')
    opt = parser.parse_args()
                                    # 參數說明
                                    # blocksize (int): 每個 patch 的邊長（正方形）。
                                    # blockstep (int): 搜尋視窗內滑動的步長。
                                    # searchsize (int): 搜尋視窗半徑，會在 (y, x) 周圍 2*searchsize 範圍內搜尋。
                                    # diffT (float): 計算兩區塊差異的門檻，控制進入 group 的相似性
                                    # coefT (float): DCT 係數的閾值，小於此值的會被設為 0


    # 讀入影像與預處理
    # ============================================================================================
    image = cv2.imread(opt.image_path)              # 讀取圖像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 轉為灰階
    image = image.astype(np.float32) / 255.0        # 正則化，imread預設是uint8，所以要除255要先轉float
    show_img(image, title='Original Image')         # 展示圖像

    # 加入高斯雜訊
    # ============================================================================================
    sigma = opt.sigma                               # 高斯雜訊標準差
    imnoise = add_gaussian_noise(image, sigma)      # 加入雜訊
    imnoise = prepare_image(imnoise, blocksize=opt.step1_blocksize,
                            blockstep=opt.step1_blockstep,
                            searchsize=opt.step1_searchsize)
                                                    # 雜訊影像進行邊界處理與尺寸對齊
                                                    # 讓影像的寬高可以剛好被 block size 與步長整除
                                                    # 避免後續 DCT 時 patch 超出邊界

    show_img(imnoise, title='Noisy Image')          # 展示圖像

    # 執行 BM3D 第一階段
    # ============================================================================================
    imbasic = bm3d_step1(imnoise, opt)       # 步驟
                                                    # Step1_Grouping（分群匹配）
                                                        # 目的：找出與目前參考 patch 最相似的其他 patch，形成 3D 群組
                                                        # 以每個 (y, x) 為中心，在 searchsize 範圍內搜尋
                                                        # 使用 L1 距離（或 L2）計算相似度
                                                        # 根據 diffT 或 Top-K 排序，選出最多 blockmaxnum 個相似 patch
                                                        # 形成一個 shape 為 (blocksize, blocksize, N) 的
                                                        # 3D patch stack（patch group）
                                                    # Step1_3DFiltering（3D 濾波）
                                                        # 在 3 維 patch group 上進行稀疏變換與濾波（降噪）
                                                        # 對每個 patch 做 2D DCT
                                                        # 對相同位置的像素堆疊（沿 z 軸）做 1D DCT
                                                        # 再對 DCT 結果做 hard thresholding（小於 coefT 的設為 0）
                                                        # 然後依序做 1D IDCT + 2D IDCT 還原空間影像
                                                    # Step1_Aggregation（加權聚合）
                                                        # 將多個重疊的 patch 還原結果加權融合，回復成完整影像
                                                        # 每個 patch 還原後，放回原影像對應位置
                                                        # 使用 Kaiser window 做空間加權（中間權重大，邊緣小）
                                                        # 將所有 patch 疊加至 numerator / denominator buffer
                                                        # 最後整張圖為 numerator / denominator

    show_img(imbasic, title='Step1 Image')        # 展示圖像
                                                    # 去除了大多數雜訊，但也會損失一些細節與高頻紋理
                                                    # 它的目的不是「直接作為輸出」，而是作為第二階段的 參考圖（pilot estimate）

    # 執行 BM3D 第二階段
    # ============================================================================================
    imfinal = bm3d_step2(imnoise, imbasic, opt)
    show_img(imfinal, title='step2 Image')