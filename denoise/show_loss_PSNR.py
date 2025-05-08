import re
import matplotlib.pyplot as plt
import os

# 假設日誌儲存在 logs/epoch1.txt 中
log_file = "logs/epoch1.txt"

# 儲存 loss 和 PSNR
losses = []
psnrs = []

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        match = re.search(r"loss:\s*([0-9.]+)\s*PSNR_train:\s*([0-9.]+)", line)
        if match:
            loss = float(match.group(1))
            psnr = float(match.group(2))
            losses.append(loss)
            psnrs.append(psnr)

# print(losses)
# print(psnrs)

# 確保 logs 資料夾存在
os.makedirs("logs", exist_ok=True)

# 繪製 Loss 圖表並儲存
plt.figure()
plt.plot(losses, label='Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Loss over Batches')
plt.grid(True)
plt.legend()
plt.savefig("logs/loss_over_batches.png")

# 繪製 PSNR 圖表並儲存
plt.figure()
plt.plot(psnrs, label='PSNR', color='orange')
plt.xlabel('Batch')
plt.ylabel('PSNR (dB)')
plt.title('PSNR over Batches')
plt.grid(True)
plt.legend()
plt.savefig("logs/psnr_over_batches.png")

# 可選：顯示圖表（非必要）
# plt.show()
