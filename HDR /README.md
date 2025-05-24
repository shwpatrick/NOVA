

# 概覽

本篇是HDR轉LDR相關技術的內容研讀與實作整理  
參考論文：[Lightness Perception in Tone Reproduction
for High Dynamic Range Images](https://resources.mpi-inf.mpg.de/hdr/lightness/krawczyk05eg.pdf)  
圖像資料：[openexr網站](https://openexr.com/en/latest/test_images/ScanLines/Desk.html)  

後續內容包含條目如下：  

- [概念筆記](#概念筆記) - HDR轉LDR概念摘要
- [實作框架拆解](#實作框架拆解) - 框架拆解實作流程(盡可能仿照論文內容)
- [圖解流程演示](#圖解流程演示) - 對於每個步驟的圖形展示(展示用)  
- [觀察研究](#觀察研究) - 對於實作內容的探討
    - [log極小值與σ對於有效框的影響](#log極小值與σ對於有效框的影響) -> log極小值直接影響到了σ的決定值
    - [k-means後合併策略對於σ選定的影響](#k-means後合併策略對於σ選定的影響) -> 設計了兩種合併策略，結果找到了一模一樣的最終中心點   	
    - [articulation的常態性失能觀察](#articulation的常態性失能情況觀察) -> articulation通常都會失效的因果關係，以及articulation生效的條件
        - [articulation為什麼會常態性失能](#articulation為甚麼會常態性失能) -> 動態範圍是兩倍的σ，而σ因為條件而一定會大於1(動態範圍大於2)
        - [articulation什麼情況下會發揮效果](#articulation什麼情況下會發揮效果) -> y_max 跟 y_min 碰到了亮度邊界，因此發生不是再是P=0.6時的數值
        - [articulation如果強迫發揮加權效果會如何](#articulation如果強迫發揮加權效果會如何) -> 合併框架中沒有被選擇合併的框架(在本圖中是過曝框架)更有可能因為加權而消失  
    - [其他實驗記錄](#其他實驗記錄)
        - [k-means的k值設置](#k-means的k值設置)
        - [採用圖像的細微差異](#採用圖像的細微差異)
        - [所有中心點獨立σ設計](#所有中心點獨立σ設計)
        - [雙邊濾波器與第二次σ選用](#雙邊濾波器與第二次σ選用)
- [總結](#總結)  
- [程式參數使用介紹](#程式參數使用介紹)
- [既有資料夾介紹](#既有資料夾介紹)
---

# 概念筆記

## 明度 vs. 亮度（Lightness vs. Luminance）

在影像處理與人眼視覺中，**明度（lightness）** 和 **亮度（luminance）** 是不同的概念：

- **亮度（luminance）**：物理量，代表實際的光強度。
- **明度（lightness）**：人眼主觀感知的「亮」程度，具有視覺恆常性（constancy）。

>明度的恆常性不是絕對的（容易失效）
>恆常性是指即使環境光改變，我們仍能感知物體「一樣亮」。

---

## 以下情況會造成明度失真或恆常性失效：

- **視野變窄**：減少參照環境，恆常性降低
- **亮度動態範圍受限**：感知空間縮小，誤判亮暗
- **場景佔據視網膜區域大**：提供穩定參照 → 恆常性提升
- **同時對比效應**：明暗受到周圍顏色影響
- **Gelb 錯覺**：明暗認知錯位（某些裝置無法重現）

## 為什麼這與顯示設備有關？

 > 人眼會根據環境自動「調整」對光的敏感度，但螢幕不會。
 
 - 顯示裝置通常亮度範圍固定，無法觸發人眼對光的動態適應
- 所以為了讓畫面在螢幕上「看起來自然」，我們需要：
  - 模擬人眼對光的反應行為
  - 對影像進行預處理(tone-mapping)

---

# 人眼調光的邏輯

螢幕無法觸發人眼的調光反應，導致明度恆常性失效，因此須預先模擬此機制以改善影像感知。

## 1. 明度知覺理論（Lightness Perception Theory）

論述：人眼對於區域的明度感知取決於該區域與鄰近區域在邊界處的亮度比例。
## 2. 明度定位理論（Anchoring Theory）

論述：人眼將絕對亮度轉換為相對明度來進行感知。
## 3. 內在影像模型

影像可視為兩個主要組成層的相乘關係：

- **照明層（illumination layer）**：表示場景光照分佈，通常為低頻、高對比的成分。
- **反射層（reflectance layer）**： 表示物體表面的固有色與紋理，屬於高頻、低對比的成分。

> 概念釐清  
> - **照明層 → 低頻** -> 沒有細節、邊緣，控制整體明暗趨勢 -> 在頻域上表現為低頻成分（變化慢） 
> - **反射層 → 高頻** -> 物體邊緣、紋理、材質變化 -> 在頻域上表現為高頻成分（變化快）

> 這裡的照明層跟反射不是指主光源區域跟散射反射光源區域的關係，更像是圖形疊加

---

# 錨框（Anchoring Framework）

綜合前述概念，整體處理邏輯是：  
將畫面在全局與局部層次進行分區分層，針對每個區塊調整其明度，再重新組合還原出整張圖像。

這引出一個核心問題：  
演算法上如何將影像劃分為適當的局部區塊，並決定每一塊的「錨點亮度」？

---

## 錨定規則（Anchoring Rules）

所謂「錨點」，指的是該區域中**被視為白色或基準亮度**的參考值。  
目的是建立**亮度與明度之間的映射關係**，讓後續 tone mapping 有據可依。

根據人眼感知行為與實驗設計，主要錨定規則如下：

- **最高亮度規則**：視野中的最高亮度區域會被視為「白色」
- **平均亮度規則**：視野中的平均亮度被視為「中灰」

> 實驗結果表明：最高亮度規則更接近實際的人眼知覺

除了亮度本身，人眼對區域面積也有明顯感知偏好：

- 最高亮度區域 -> 傾向視為白色
- 最大面積區域 -> 傾向視為白色

綜合以上的概念可以提出一個整體性的結論

- 視野分區最大面積如果是 最高亮度區域 -> 是一個穩定的錨點（明度感知一致）
- 視野分區最大面積如果是 較暗亮度區域 -> 人眼會把畫面中的亮區視為自發光的光源（明度恆常性失效）

> 錨點不是單一亮度，而是加權平均亮度(面積比例加權)

---

## 複雜影像中的多框架錨定：突破單一錨點映射的侷限

單一錨點無法充分描述現實影像中的多重光源與照明條件。  
特別是在存在陰影、局部背光或高反差區時，固定一個錨定點會導致明度失真。

為了對應這種多樣光照環境，影像需分解為多個子區域，稱為 **框架（framework）**。  
每個框架對應一組相似照明條件下的像素，分別建立錨點估計。

但與傳統「一框一錨」不同，[Paper](https://resources.mpi-inf.mpg.de/hdr/lightness/krawczyk05eg.pdf)進一步引入**錨框疊加模型**：

- 每個像素可能同時屬於多個框架
- 各框架對像素的明度貢獻由機率圖與 articulation 加權
- 明度映射不再是單點參照，而是加權融合

此設計能更精確地模擬人眼在多光源環境中的亮度感知行為。

---

# 實作框架拆解

## 框架拆解流程(精簡)

step 1. 讀取圖片，計算亮度，並取log  
step 2. 以間距1為條件找出log明度範圍涵蓋間距(作為k-mean 初始中心點)  
step 3. 使用k-mean 計算中心點直到收斂  
step 4. 
	- 去除沒有對應像素的中心點  
	- 合併相鄰中心點距離1以內的中心點  
	- 計算機率圖  
	- 使用機率圖 P>0.6 確認像素對應的有效框架  
	- 計算框架articulation  
	- 使用articulation 對框架加權  
	- 使用P>0.6 重算有效框架  
	- 合併無效框架直到僅剩下有效框架  

---

 ## 框架拆解流程細述

 本流程根據論文《Lightness Perception in Tone Reproduction for High Dynamic Range Images》實作步驟重構  
 目的是將 HDR 影像分解為多個照明框架，模擬人眼明度感知過程  

### Step 1. 圖像預處理

- 讀入 HDR 圖像
- 計算 luminance（亮度）：
  $Y = 0.2126 R + 0.7152 G + 0.0722 B$
- 轉為對數空間：
  $\log_{10}(Y)$

### Step 2. 初始化框架中心點

- 根據 log 亮度範圍，**每 1 log 單位**初始化一個 K-means 中心點
- 此初始化策略保證每個中心的亮度差異不小於 1

### Step 3. K-means 分群至收斂

- 在 log 亮度空間中執行 K-means clustering
- 得到初步框架（framework）的中心點

### Step 4. 框架精練（Refinement）

- 4.1 移除空框架 - 移除未涵蓋任何像素的中心點（空群）
- 4.2 合併鄰近中心點 - 若任兩中心點距離 < 1 log 單位，進行加權平均合併  

### Step 5. 機率圖與有效框架檢查

- 5.1 計算高斯機率圖 → 對每個框架中心 $\( C_i \)$ 計算：
  $P_i(x, y) = \exp\left( -\frac{(C_i - Y(x, y))^2}{2\sigma^2} \right)$  
  σ 為所有相鄰中心點間的最大距離
- 5.2 定義有效框架區域 → 
  當某 pixel 對框架 $i$ 的機率 $\( P_i > 0.6 \)$，視為屬於該框架
- 若一個框架沒有任何 pixel 滿足此條件 → 移除

### Step 6. 框架連結強度（Articulation）

對每個有效框架
$i$
計算其動態範圍：
$\Delta Y_i = \max(Y_i) - \min(Y_i)$  
對應 articulation：
$A_i = 1 - \exp\left( -\frac{\Delta Y_i^2}{2 \cdot \theta^2} \right), \quad \theta \approx 0.33$

### Step 7. 機率加權與重正規化

- 對每個框架機率圖乘上對應
$A_i$  
- 對每個像素沿框架軸 normalize，使總和為 1

### Step 8. 框架最終篩選與合併

- 使用加權後機率圖再次檢查有效區域（P > 0.6）
- 合併仍無有效區域之框架
- 若有中心點距離過近（< 1），再度合併並重計

---

# 圖解流程演示

論文的核心概念是HDR圖片由於顯示器設備問題，很難觸發人眼的自動調亮機制  
所以在顯示器顯示圖片時，對於HDR圖像進行預調亮(Tone Mapping)，再進行圖像顯示  
做法是將圖片分成不同的光照區塊，對於不同區塊進行亮度調整  
模仿人眼機制處理圖像，讓人眼看到更接近自然調亮的顯示圖片  

## 載入圖像
目前顯示的是clipped後的圖片(線性處理)  
可以看出來明顯過曝  
暗部區域也看不出紋理細節  
這也就是論文設計的演算法的處理目標  

<p align="left">
  <img src="https://github.com/user-attachments/assets/5c7d5830-ea14-404a-9d59-72c65b589d92"  width="30%">
</p>


## K-means
目的：為了將光照區塊(後續稱為框架)進行分群  
實作：按照亮度取log後的間距，分布初始中心點數量  並使用K-means自動收斂  
![k-mean](https://github.com/user-attachments/assets/f621bd24-da60-4cb4-967a-51a6f2dd9905)

## 刪除空的框架  
實作：將圖片中空的框架進行刪除  
補充：從圖片看起來沒有需要(可能是為了某些特殊條件圖片的需求)  

![delete_empty](https://github.com/user-attachments/assets/5546d7f0-45bd-4873-8de5-8d7898708ed4)

## 合併框架
目的：將區別過近的圖區視為同一個光照條件的區域，進行合併  
實作：合併中心點相鄰距離<1的中心點群組  

![Greedy_Linear](https://github.com/user-attachments/assets/f25bbe7c-e4ce-40a9-beb6-10e03630b395)

## 計算高斯σ
目的：對於每個像素是否屬於框架，使用的是軟性的分配而不是硬性的分群，所以需要計算機率  
實作：計算σ並且配置高斯分布  

![Merged_with_sigma](https://github.com/user-attachments/assets/637df8b0-ee73-4913-b6c7-030740ea0863)

## 計算機率圖
目的：使用機率條件設置每個像素是否屬於這個框架，可信度過低的像素會被濾除  
實作：顯示對於每個框架P>0.6的區塊  

![Probability_Map](https://github.com/user-attachments/assets/65c603f7-a370-4634-9fe9-c5e2c8ece557)

## 正規化
目的：  
希望每個像素屬於該框架的信心更高  
如果像素在其它框架的機率比較低，那該像素就更有可能屬於該框架  
如果像素在其它框架的機率比較高，那該像素就更不可能屬於該框架  
實作：將每個框架進行正規化，使每個像素對於框架的總機率等於1，顯示對於每個框架P>0.6的區塊  

![Probability_Map_After_Norm](https://github.com/user-attachments/assets/d86e7b29-ab1a-4c67-8cd0-0588a2acdbf1)

## 合併空框架
目的：  
前一個步驟會讓有些框架無效，將這些框架與接近的框架進行合併  
觀察Cluster Map可以知道，其實就是將淺灰色的區塊分配給亮區或暗區(結果選暗區)  
實作：  
合併P>0.6沒有像素點的框架，重新計算機率圖  
補充：  
這裡機率圖σ使用的是1.1，不再是相鄰中心距離點  
因為合併後的的中心點較少，相鄰中心距離變得很大  
如果高斯又過於平坦，可能會導致雙邊濾波的時候讓邊界又過度模糊化  
這部份Paper似乎沒有特別描述，是我自己的猜測與調整  

![Final_refine_centroids](https://github.com/user-attachments/assets/49f18105-7cac-4dc9-bdda-bf64c3eb6fb2)


## 雙邊濾波
目的：  
前面的圖片使用的分區只考慮亮度，而沒有考慮紋理  
簡單來說，對於一個物體，可能會有高光跟暗部  
只按照亮度分區就會導致高光被分配到亮的區塊  
讓一個物體的物件不再是屬於同一個框架  
從圖片上來看就是那本書的反光太亮了，但它希望書都屬於一個框架  
所以把圖片模糊平衡，這樣重新分區就會讓書都屬於暗部框架  
實作：  
對於每個框架的機率圖做雙邊濾波  
補充：  
雙邊濾波帶有將紋理模糊的成分，所以如果原本的框架分界不夠清楚  
可能會因為這個模糊動作，讓框架分界失效  
這個會在後續觀察中被討論  

每個框架對應的機率圖(雙邊濾波前)  


<p align="left">
  <img src="https://github.com/user-attachments/assets/b00cd034-4907-4dcd-90a2-3c4d32d60e08"  width="30%">
  <img src="https://github.com/user-attachments/assets/b93ea86e-5f98-4d75-8c11-3b003b943c78"  width="30%">
</p>

每個框架對應的機率圖(雙邊濾波後)  

<p align="left">
  <img src="https://github.com/user-attachments/assets/13161c05-1de7-4b86-aee8-3fe13714124b"  width="30%">
  <img src="https://github.com/user-attachments/assets/ac9fd5e0-4a14-48a0-b7ea-d73c8db349ef"  width="30%">
</p>

雙邊濾波後的機率圖(P > 0.6)  

![Bilateral_Filter](https://github.com/user-attachments/assets/fcda7263-0854-4092-a16f-7f8047ad285b)

## 選染機率圖
對於每個像素，選取機率更高的框架作為歸屬，進行渲染
<p align="left">
  <img src="https://github.com/user-attachments/assets/d6861205-2d9f-485c-81b9-56a886c2273e"  width="30%">
</p>

## 渲染機率圖(乘上Articulation加權)
對於機率圖，乘上連結強度articulation  
選取機率更高的框架作為歸屬，進行渲染(因為這裡articulation加權都是1，看不出差別)    
<p align="left">
  <img src="https://github.com/user-attachments/assets/4c828e8a-73c1-4fa2-8131-eba94b50b721"  width="30%">
</p>

## 渲染機率圖(正規化後)
在cluster_map看不出來差別  
<p align="left">
  <img src="https://github.com/user-attachments/assets/965b02fa-0805-4a32-98f0-59f4dc8ef90e"  width="30%">
</p>



---


# 觀察研究

## log極小值與σ對於有效框的影響

在實作中對於亮度取log 的時候，會做數值穩定化處理（numerical stabilization），而將極小值設置為一個下限。可以在論文的圖片中看到這個極小值設置(約1e-3.5)。
按照k-mean的設計，對於這樣的離群組在收斂後，最左側的收斂中心點會大幅度離群，而在後續合併中心點組合的設計中，由於相鄰距離超過了1，所以不會與其他中心點合併。
我們可以在paper的中圖觀察到這個現象。

![image](https://github.com/user-attachments/assets/8efb0a16-920d-4ed6-b336-b7ced2d239c8)

以下是我調整極小值設置後所計算的k-mean結果，三張圖的極小值設置分別是1e-4, 1e-3.5, 1e-3  
左圖是對於每個中心點找出框架的分群渲染，中圖是中心點分布以及亮度的histogram，右圖則是中心點對應的向素數量  

<p align="left">
  <img src="data/1e-4 - auto2.20 - greedy/k-mean.png" alt="1e-4 sigma:2.20 k-mean" width="100%">  
  <img src="data/1e-3.5 - auto1.70 - greedy/k-mean.png" alt="1e-4 sigma:1.70 k-mean" width="100%">
  <img src="data/1e-3.0 - auto1.51 - greedy - a_para 0.33/k-mean.png" alt="1e-4 sigma:1.70 k-mean" width="100%">
</p>

極小值因子在後續中帶來了極大的影響，考慮機率圖算法  
$P_i(x, y) = \exp\left( -\frac{(C_i - Y(x, y))^2}{2\sigma^2} \right)$  
σ 為所有相鄰中心點間的最大距離，極小值的離群程度幾乎直接決定了σ 的對應大小，在此整理成表格

| 極小值數值  | σ      |
| ---------- | ------ |
| 1e-4       | 2.20   |
| 1e-3.5     | 1.70   |
| 1e-3       | 1.51   |


下面的圖片組是機率圖的高斯計算，可以看得出來，σ 的數值增加使得高斯分布變得平滑  
在加權後計算正規化機率時，有效框架標準(P > 0.6)更難達到，導致沒有足夠的有效框可以進行合併  


<p align="left">
  <img src="data/1e-4 - auto2.20 - greedy/Merged_with_sigma.png" alt="1e-4 sigma:2.20 k-mean" width="100%">  
  <img src="data/1e-3.5 - auto1.70 - greedy/Merged_with_sigma.png" alt="1e-4 sigma:1.70 k-mean" width="100%">
  <img src="data/1e-3.0 - auto1.51 - greedy - a_para 0.33/Merged_with_sigma.png" alt="1e-4 sigma:1.70 k-mean" width="100%">
</p>

下圖是進行正規化後，對於每個中心點組成的框(P>0.6)的條件下，達到標準的像素分布  
而為空的框則會在下次被鄰近的有效框合併  
但在這些極小值設置下，剩於有效框可能會低於兩個
導致後續的框架合併無法正常運作

<p align="left">
  <img src="data/1e-4 - auto2.20 - greedy/Probability_Map_After_Norm.png" alt="1e-4 sigma:2.20 k-mean" width="100%">  
  <img src="data/1e-3.5 - auto1.70 - greedy/Probability_Map_After_Norm.png" alt="1e-4 sigma:1.70 k-mean" width="100%">
  <img src="data/1e-3.0 - auto1.51 - greedy - a_para 0.33/Probability_Map_After_Norm.png" alt="1e-4 sigma:1.70 k-mean" width="100%">
</p>

至此我開始思考 σ 設置對於有效框判定的影響，我改將 σ 設置為固定的常數而非最大相鄰中心點距離  
以下分別將 σ 設置成1.1 以及0.5 ，使用極小值1e-4  

1.1的數值選定，是為了符合原本相鄰中心距離必須 > 1否則合併中心的假設，這個是極度符合條件的最小值  
0.5的數值選定，則是想藉此反向觀察，當 σ 小於 1 時違反了中心距離的條件，也就是相鄰中心距離必須 > 1 時的意義為何？  
1e-4的極小值選定則是因為，這個數值在原本的設置中會全部框架不符合 P > 0.6 的條件  
我想以此觀察將極小值與 σ 的關聯性分離會產生的變化  

> 影響框架合併效果的是極小值設置還是 σ 設置？

以下四張圖片分別代表 

- σ = 1.1, 正則化前  
- σ = 1.1, 正則化後  
- σ = 0.5, 正則化前  
- σ = 0.5, 正則化後  

<p align="left">
  <img src="data/1e-4.0 - hard1.10 - greedy/Probability_Map.png" alt="1e-4 sigma:2.20 k-mean" width="100%">  
  <img src="data/1e-4.0 - hard1.10 - greedy/Probability_Map_After_Norm.png" alt="1e-4 sigma:1.70 k-mean" width="100%">
  <img src="data/1e-4.0 - hard0.50 - greedy/Probability_Map.png" alt="1e-4 sigma:1.70 k-mean" width="100%">
  <img src="data/1e-4.0 - hard0.50 - greedy/Probability_Map_After_Norm.png" alt="1e-4 sigma:1.70 k-mean" width="100%">
</p>
  
我們可以藉此得出一個結論  
我們必須要有足夠大的 σ ，才可以讓中間的數個框架受到左右相鄰的高斯分布正則化加權影響而消失，被判為無效框架  
我們也需要有足夠小的 σ ，否則過於平坦的高斯分布可能會單靠單邊就使得最左或最右的高斯正則化加權機率低於0.6，導致所有框變得無效  

即便極小值設置的更小，調低 σ 仍舊能讓有效框架超過兩個
意即，有效框架選定主要受 σ 影響  
而這個 σ 如果是由最大相鄰中心距離決定，間接變成了極小值的位置決定了最後框架合併的效果

---

## k-means後合併策略對於σ選定的影響

前文討論了 σ 選定對於最終機率分布的影響  
在實作時我產生了額外的疑問：

> 不同的合併策略，可不可以導致合併後的最大相鄰中心距離不再是極小值中心點的相鄰距離？

對此進行了兩種合併算法的設計
1. greedy linear :  
  左到右遍歷中心點群組，當目前中心點與右側群組距離小於 1 時，進行合併  
  合併會持續向右進行，直到遇到距離 ≥ 1 的中心點為止，然後從下一個未處理的中心點繼續    
3. closest first :  
  計算目前相鄰中心點群組的距離，選出最短距離的相鄰中心點群組，進行合併  
  重複這個動作直到所有相鄰中心點群組大於 1  

而實作後我意外地發現，對於1e-4、1e-3.5、1e-3 這三個極小值設定產生的k-means中心分布  
不論用greedy linear或者是使用closest first合併方法  
兩者找到的中心點群組都一模一樣  

而這個可能不是巧合  
觀察1e-4的k-means後中心組[-3.79, -1.94, -1.37, -0.68, -0.01, -0.79, -1.69]  
其對應產生的相鄰間距分別是[1.85, 0.57, 0.69, 0.70, 0.78, 0.90]  
除了最左側的極小值相鄰中心點間距已經超過了1，其它的相鄰中心點間距剛好遞增排列  

推測，可能是k-means的迭代收斂特性自動導向了這個結果  
重新思考k-means的設計  
初始中心點是取log後均勻分布間距為1的中心點  
而在進行收斂迭代時，最左側的極小值實際上是被壓縮過的聚合，會有極大的權重  
這導致迭代動作，就是其它的中心點受到擠壓而向右移動  
根據受到擠壓的程度不同而變成了距離1以下的遞增數組  

在這個情況下，linear greedy以及closest first算法  
兩者都是從第二個centroid開始逐步向右合併  
所以會得出一模一樣的最終中心點分布  

因為最左中心點間距會極大於1，而其它剩餘中心點則會逐漸合併直到間距大於1  
這也可以佐證上一個結論，極小值的設置值在σ為「最大相鄰中心距離」時   
變相地直接決定了σ數值

---

## articulation的常態性失能情況觀察

Paper中提及了articulation(連結強度)在通常情況下都是無效狀態  
在進行實作時，我也注意到每個框架的articulation計算基本上都是1  
這導致進行加權時，每個框架加權都相同，整個articulation的設計處在常態性失能的狀態  
雖然可以理解為這個是面對某些極端條件才會有效觸發的特殊參數，但我會想知道：

> articulation為甚麼會常態性失能？
> articulation什麼情況下會發揮效果？
> articulation如果強迫發揮加權效果會如何？

### articulation為甚麼會常態性失能

articulation 由框架中的動態範圍所決定，而目前的算法推演會導致articulation失能  
在此進行數學推導：

在初始框架分群後，我們使用 $P > 0.6$ 作為機率圖的有效像素門檻。

對於單一框架，其中心點為 $C_i$，若該框架的機率分布不受邊界裁切影響，且高斯機率為對稱分布，我們可設：

- $Y_{\max} = C_i + \text{diff}$
- $Y_{\min} = C_i - \text{diff}$

並根據高斯機率圖條件：

$$
P_i(Y_{\max}) = P_i(Y_{\min}) = \exp\left(-\frac{(C_i - Y)^2}{2\sigma^2}\right) = 0.6
$$

代入 $Y = C_i \pm \text{diff}$：

$$
\exp\left(-\frac{\text{diff}^2}{2\sigma^2}\right) = 0.6
\Rightarrow
-\frac{\text{diff}^2}{2\sigma^2} = \ln(0.6)
\Rightarrow
\frac{\text{diff}}{\sigma} = \sqrt{-2 \ln(0.6)} \approx 1.0108
$$

因此：

$$
\text{diff} \approx 1.0108 \cdot \sigma
$$

整體動態範圍為：

$$
Y_{\max} - Y_{\min} = 2 \cdot \text{diff} = 2.036 \cdot \sigma
$$

在論文中的中心點 refine 後，我們保證所有相鄰中心點距離 $\geq 1$，因此：

$$
\sigma \geq 1.0 \Rightarrow \text{Dynamic range} \geq 2.036
$$

articulation 定義如下：

$$
A_i = 1 - \exp\left(-\frac{(\Delta Y_i)^2}{2 \cdot 0.33^2} \right)
$$

代入 $\Delta Y_i = 2.036$：

$$
\frac{(2.036)^2}{2 \cdot 0.33^2} = \frac{4.15}{0.2178} \approx 19.06
\Rightarrow
A_i \approx 1 - \exp(-19.06) \approx 1.0
$$

在此做一個小結，當 σ 大於 1 的設計，初始有效框架的選定為 P > 0.6  
就會直接導致articulation 失能的結果  

### articulation什麼情況下會發揮效果

從上面的推論繼續向下延伸，在正常的高斯分布下：

- $Y_{\max} \approx C_i + 1.0108 \cdot \sigma$
- $Y_{\min} \approx C_i - 1.0108 \cdot \sigma$
- $dynamic range \approx 2.0216 \cdot \sigma$

![image](https://github.com/user-attachments/assets/ae57dcd0-dfb9-446a-833b-df8bc79bc9c9)  

按照這張圖片的動態範圍與articulation的對應關係  
除非 σ 數值極小(但這個可能性已經在這個演算法設計中被排除)  
或者 $Y_{\max}$ 或 $Y_{\min}$ 受到亮度邊界裁切而不再是高斯P=0.6的數值  
articulation 才會因為動態範圍變小而發生作用  

## articulation如果強迫發揮加權效果會如何

從Paper的圖片描述動態範圍與articulation 的關係  
我們可以看出動態範圍在0.5以下的情況下articulation 才會生效  
而如果使用(P > 0.6)的框架，其動態範圍基本上很難達成要求

重新觀察articulation公式：

$$
A_i = 1 - \exp\left(-\frac{(\Delta Y_i)^2}{2 \cdot 0.33^2} \right)
$$

我們可以看到除了動態範圍外，還存在參數0.33(後續稱為a_para)  
我是否可以透過調整a_para的方法，讓articulation低於1，達到真正具備加權效果的影響？  
因此我做了一組對照組與實驗組，分別為a_para = 0.33 以及 a_para = 2.0  
2.0 這個數字是手動慢慢調整的，將articulation降到可以看到不同加權的程度  

圖形方面則是一樣使用1e-3以及1e-4分別作圖  
以下則是對應每個框架的articulation 實際參數  

1e-3 auto1.51 apara 0.33  
mask idx: 0 articulation: 1.0 , min: -3.0 , max: -0.1186158 , diff: 2.8813841  
mask idx: 1 articulation: 1.0 , min: -1.9590952 , max: 1.090032 , diff: 3.049127  
mask idx: 2 articulation: 1.0 , min: -0.45076218 , max: 2.3047442 , diff: 2.7555065  

1e-3 auto1.51 apara 2.00  
mask idx: 0 articulation: 0.6457657 , min: -3.0 , max: -0.1186158 , diff: 2.8813841  
mask idx: 1 articulation: 0.6871862 , min: -1.9590952 , max: 1.090032 , diff: 3.049127  
mask idx: 2 articulation: 0.6129115 , min: -0.45076218 , max: 2.3047442 , diff: 2.7555065  

<p align="left">
  <img src="data/1e-3.0 - auto1.51 - greedy - a_para 0.33/Probability_Map_After_Norm.png"  width="100%">
  <img src="data/1e-3.0 - auto1.51 - greedy - a_para 2.00/Probability_Map_After_Norm.png"  width="100%">
</p>

1e-4 auto2.20 apara 0.33  
mask idx: 0 articulation: 1.0 , min: -4.0 , max: -1.5650234 , diff: 2.4349766  
mask idx: 1 articulation: 1.0 , min: -3.8081334 , max: 0.63550997 , diff: 4.4436436  
mask idx: 2 articulation: 1.0 , min: -2.6387553 , max: 1.8096563 , diff: 4.4484115  
mask idx: 3 articulation: 1.0 , min: -1.1466537 , max: 2.3047442 , diff: 3.451398  

1e-4 auto2.20 apara 2.00    
mask idx: 0 articulation: 0.52342916 , min: -4.0 , max: -1.5650234 , diff: 2.4349766  
mask idx: 1 articulation: 0.91526663 , min: -3.8081334 , max: 0.63550997 , diff: 4.4436436  
mask idx: 2 articulation: 0.9157145 , min: -2.6387553 , max: 1.8096563 , diff: 4.4484115  
mask idx: 3 articulation: 0.774406 , min: -1.1466537 , max: 2.3047442 , diff: 3.451398  

<p align="left">
  <img src="data/1e-4.0 - auto2.20 - greedy - a_para 0.33/Probability_Map_After_Norm.png"  width="100%">
  <img src="data/1e-4.0 - auto2.20 - greedy - a_para 2.00/Probability_Map_After_Norm.png"  width="100%">
</p>

首先解釋，這裡的articulation計算是在第一次計算高斯機率圖後  
但實際上的articulation 應該計算在第二次計算高斯機率圖，且經過雙邊濾波器之後  
但在這裡由於有多個框架更可以看得出來articulation對於圖片的影響方法  

從這些數據來看我們可以觀察到，調整a_para確實可以讓articulation加權權重生效  
整體而言是讓極左跟極右的圖形因為受到裁切而權重降低  
經歷過合併的框架會在log亮度中更接近中心點，也就更不易被裁切  
也就是說，在合併框架中沒有被選擇合併的框架(在本圖中是過曝框架) 更有可能因為加權而消失  

---

# 其他實驗記錄
   
## k-means k值設置

實驗後注意到了paper的k-mean初始中心群  
其實是對log上下都取整再去計算k的數值  
所以與我的中心群數量有所不同  
較高密度的centroids也影響到了自動計算σ的數值  
所以Paper在同樣的極小值計算下，σ會有比較小的數值  
以下依序分別是在1e-3 選取 k = 8(Paper數量), 6的結果    

更高密度的K值設計固然會讓合併框架更平滑合理  
但也會更容易讓兩個極值有效框受到過近的擠壓而在正則化後失效  
是個值得注意與考慮的點  

<p align="left">
  <img src="https://github.com/user-attachments/assets/fb227ff4-3985-4688-afa6-e8cc73358054"  width="100%">
  <img src="https://github.com/user-attachments/assets/ee14d20d-6c3b-40d4-b357-1d8ac45a2e63"  width="100%">
</p>

![Probability_Map_After_Norm](https://github.com/user-attachments/assets/6a2fd4d0-cf1e-4cd5-9ffe-85935ef1bd13)
![Probability_Map_After_Norm](https://github.com/user-attachments/assets/9e6525da-0390-429f-b943-2d2f78da2ab9)

## 採用圖像的細微差異

觀察paper跟我的分布圖可以發現  
我的圖片跟Paper圖片在最高亮度區域有所不同  
我的實際上少了一截，(可能由於載入方法或者是圖源不同)  
因此實作上即使都使用Paper提供的參數  
效果可能也有一定的落差  

<img src="https://github.com/user-attachments/assets/ac72d4a9-ebd0-4779-a0c0-655b09387536" style="width: 50%;">
<img src="https://github.com/user-attachments/assets//bc5ba21f-8f5f-4767-a56f-e24581b8f899" style="width: 50%;">

## 所有中心點獨立σ設計

認知到σ對於區分影響極大後，我開始思考對於每個centroid該做不同的σ  
讓加權的效果有所不同，實現在程式裡的auto_multi方法  
使用的是單個cetroid對於相鄰左右的centroid取較大值作為 σ  
某種程度上來說更容易保留極左極右框架  
但沒有更繼續的深入研究與比較  

![Merged_with_sigma](https://github.com/user-attachments/assets/88ec62a6-8f38-4b24-a658-407f8022491c)

## 雙邊濾波器與第二次σ選用

雙邊濾波器主要在於平滑紋理  
但同時，如果機率圖σ過大->高斯過於平坦  
很容易使得雙邊濾波後框架間的分界也隨之消失  
所以我比較了直接設置固定值與計算值的效果  
這邊依序採用了三種第二次機率的數值1.11(hard), 1.51(hard), 2.19(auto)  
我會更傾向於使用固定數值分界，但這個可能會在使用其他圖片時，得不到同樣結論  

<img src="https://github.com/user-attachments/assets/dce5c8f2-01bc-4226-97fd-11252818d28b" style="width: 30%;">
<img src="https://github.com/user-attachments/assets/d8375eb4-77ec-47bd-a486-b475749ac9ca" style="width: 30%;">
<img src="https://github.com/user-attachments/assets/d4e1bf23-d2da-45c7-aa68-928c23842860" style="width: 30%;">

---

# 總結

經過了這些實驗測試，我覺得這是一個很tricky的作法  
透過高斯函數左右夾擊讓中間的框全部都變成無效框  
又要讓單邊的高斯不會過度影響到極值的兩個框架，導致任何之一無法存活  

σ 在這個方法裡扮演了十分重要的決策因子  
但與極小值選取的關聯性讓我不太能接受  
雙邊濾波器目前似乎也不太受 σ 影響而進行調整  

如果要讓這個框架可以自適應於所有影像而不用手動調參  
極小值與雙邊濾波器，甚至是有效框架的判定或者聚合方法值得更多的討論  

雙邊濾波在其中占據了大量的計算資源  
而回頭看初次機率圖σ=1.1的設計  
最右側中心集群P>0.6的遮罩似乎就是最佳的目標錨框

說不定簡易的二分法找出過曝光源
套用固定σ=1並選取最右側光源作為發光體錨框
會是一個更快速有效的方法  

---

# 程式參數使用介紹

    # ============================================================
    black_squeeze = -3            # 極小值設置
    sigma_mode = 'auto'           # hard, auto, auto_multi
          # hard : 使用固定數值
          # auto : 自動計算相鄰中心點最大距離
          # auto_multi : 對於每個centroid 配置獨立sigma，數值為左右相鄰centroid中的較大值
    hard_sigma = False            # 是否對於第一次機率圖使用hard sigma
                                  # 如果使用auto 或者auto_multi請設為false
    hard_sigma_value = 0.5        # hard sigma數值
    # ============================================================
    strategy = 'greedy'           # 合併策略
        # closest : closest first(論文作法)
        # greedy : greedy linear
    # ============================================================
    second_hard_sigma = False     # 第二次機率圖sigma計算
    second_hard_sigma_value = 1.1
        # 如果使用false 則會沿用第一個sigma的策略
        # 第一次選hard會套用hard sigma 的數值
    # ============================================================
    a_para = 0.33 # articulation參數，論文設置0.33
    arti_mode = 1
        # 0 : 不計算articulation
        # 1 : 計算articulation 使用P>0.6(論文作法)
        # 2 : 計算articulation 使用centroid硬分界決定y_max, y_min
    bila_mode = 1
        # 0 : 不套用bilateral filter
        # 1 : 套用bilateral filter
    kmean_mode = 1
        # 0 : 取log後直接計算間距，k會比論文作法小
        # 1 : 取log後，對於上下界向上下取整，k會比較大(論文作法)
    # ============================================================
    base_folder = "data_changekmeank"
        # 程式會依照 極小值 - 第一次sigma策略與數值 - 合併方法 - a_para
        # 建立對應資料夾於base_folder之下
        # 由於auto 方法的sigma 不會在初始計算，因此一開始會設置為 autoUNKNOWN
        # 後續在確定sigma數值後更改資料夾名稱
        # 如果有跟更改後的資料夾名稱同名的資料夾，會直接把舊資料夾刪除
        # 如果要進行其它的測試，請手動更改資料夾名稱區分資料
    
# 既有資料夾介紹

- data_changekmeank : 改用了論文的k-means的k取法
- data_v4 : 測試prob計算的articulation a_para 與 bilateral 的關係
- data_v5 ：套用全新的articulation 設計 也就是以cetnroids 為核的實驗(但其實articulation都不太生效)



<script type="text/javascript"
  async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
