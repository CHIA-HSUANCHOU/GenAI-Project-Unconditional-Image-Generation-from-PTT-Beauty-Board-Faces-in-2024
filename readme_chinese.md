
# Unconditional Image Generation from PTT Beauty Board Faces in 2024

---
Project: Unconditional Image Generation from PTT Beauty Board Faces in 2024

Author: CHOU CHIA-HSUAN

Date: 2025-05-27

Course: Generative AI

---

## 1. Data Structure
<pre> 
├── scripts/       # 爬蟲、前處理、模型訓練等程式碼與相關設定檔
    ├── saveimagelinkmonth.py   # 擷取 PTT 表特板圖片連結並按月份分類儲存
    ├── run_all_parallel_threaded.py  # 多線程控制：呼叫 downloader 處理每月任務
    ├── download_worker.py # 實際下載圖片並進行臉部與清晰度初步篩選
    ├── faceposition.py  # 圖片預處理（對齊、裁剪、模糊、分類）
    ├── diffusion.py # Generation image with diffusion model
    ├── Diffusion_submit.ipynb # Generation image with diffusion model
├── README.md                 

</pre> 

* 環境：

saveimagelinkmonth.py, run_all_parallel_threaded.py ,download_worker.py:本機端跑

faceposition.py, diffusion.py, Diffusion_submit.ipynb ：colab訓練(將爬蟲抓到的圖片傳到google colab)

## 2. Crawl
`saveimagelinkmonth.py`
1. 找出**從 2024 年 1 月 1 日至 12 月 31 日的 PTT 表特板文章**範圍，擷取每篇文章中的所有圖片連結（jpg/png/gif 格式），並將資料儲存為 image_links.jsonl。

2. 將擷取結果依照月份分類為 image_links_01.jsonl、image_links_02.jsonl……，存入 monthly_jsonl/ 資料夾中。

`run_all_parallel_threaded.py, download_worker.py`

3. 使用 ThreadPoolExecutor(max_workers=4) 同時平行執行最多 4 個月份的圖片下載任務，主控程式會依序呼叫 download_worker.py 處理每個月的 JSONL 資料。

4. 在每張圖片下載後，透過 OpenCV 的 DNN 模型進行臉部偵測，僅保留含有人臉的圖片；再計算 Laplacian 變異數判斷圖像清晰度（閾值為 60），只儲存「**同時有臉部且清晰**」的圖片。

## 3. Preprocessing
`faceposition.py`

1. 圖像品質檢查

將每張彩色圖轉灰階，計算 Laplacian variance 作為清晰度，只要清晰度大於門檻(60)的圖片。

2. 臉部偵測與對齊

使用 InsightFace 的 app.get() 函式對圖像進行臉部偵測，僅保留最大面積的人臉。若無法偵測到臉部，或偵測結果中不含有效的五官關鍵點，或人臉在原圖中所佔面積低於 1.5%，則跳過該張圖像。

若偵測成功，根據左眼與右眼座標計算兩眼連線的傾斜角度，並以兩眼中心作為旋轉中心，使用 cv2.getRotationMatrix2D() 建立仿射旋轉矩陣，再透過 cv2.warpAffine() 將原圖旋轉對齊，使**雙眼呈水平排列**。為避免旋轉時裁邊，輸出設為原圖的 1.2 倍。

3. 二次偵測與裁剪

在對齊後的圖像中再次偵測臉部，若成功，則以對齊圖中的新臉框進行裁切；若偵測失敗，則回退至原圖中的最大臉框。**裁剪時會以臉框中心為基準，擴張邊界以涵蓋整張臉部（含額頭與下巴）**。接著在裁剪後的區域內製作一個半徑為臉部寬高中最大者的 0.55 倍的圓形 feather mask，保留臉部中央區域，同時將外圍區域以高斯模糊處理進行平滑融合。最後將融合後的圖像 resize 成固定尺寸（60×60）。

4. 姿態與性別分類

根據人臉的水平旋轉角度 yaw 將臉部分類為： 
* frontal（正臉）：|yaw| < 20 度
* left（左臉）：yaw < -20 度
* right（右臉）：yaw > 20 度

再根據模型對人臉性別的預測進行標記：
* 若性別為 1 → Male
* 若性別為 0 → Female
(若模型未能辨識性別 → 當作 Female 處理)

5. 最終使用資料

原本將正臉、左側臉、右側臉分開訓練，但在統計後發現，左側臉與右側臉各僅約有 4,000 張，資料量相對不足，為避免樣本不足導致訓練不穩，最終選擇將所有姿態、性別類別合併訓練。**我手動過濾了側臉中僅露出單眼、五官不完整者，以及明顯模糊或無法辨識臉部細節的圖片**。經過清洗後的圖片即為 data/ 資料夾中的最終訓練資料，共包含正臉與可辨識的側臉樣本，作為 diffusion 模型訓練使用。



## 4. Generation process
`diffusion.py`

### 1. Unet 模型架構
本模型是以 U-Net 為基礎，設計一個用於 DDPM 的影像去噪網路。

<pre> StrongerUNet Architecture ──────────────────────────────────────────────────────────── 
Input (x, t)
   │
   ▼
DoubleConv (in: 3 → 128)
   │
   ▼
Down Block 1
  ├─ MaxPool
  ├─ DoubleConv (128 → 128, residual)
  ├─ DoubleConv (128 → 256)
  └─ + Time Embedding (t)
   │
   ▼
Self-Attention (256, size=32)
   │
   ▼
Down Block 2
  ├─ MaxPool
  ├─ DoubleConv (256 → 256, residual)
  ├─ DoubleConv (256 → 384)
  └─ + Time Embedding (t)
   │
   ▼
Self-Attention (384, size=16)
   │
   ▼
Down Block 3
  ├─ MaxPool
  ├─ DoubleConv (384 → 384, residual)
  ├─ DoubleConv (384 → 512)
  └─ + Time Embedding (t)
   │
   ▼
Self-Attention (512, size=8)
   │
   ▼
Bottleneck
  ├─ DoubleConv (512 → 768)
  ├─ DoubleConv (768 → 768)
  └─ DoubleConv (768 → 512)
   │
   ▼
Up Block 1
  ├─ Upsample
  ├─ Concat skip from Down2 (384)
  ├─ DoubleConv (896 → 896, residual)
  ├─ DoubleConv (896 → 384)
  └─ + Time Embedding (t)
   │
   ▼
Self-Attention (384, size=16)
   │
   ▼
Up Block 2
  ├─ Upsample
  ├─ Concat skip from Down1 (256)
  ├─ DoubleConv (640 → 640, residual)
  ├─ DoubleConv (640 → 256)
  └─ + Time Embedding (t)
   │
   ▼
Self-Attention (256, size=32)
   │
   ▼
Up Block 3
  ├─ Upsample
  ├─ Concat skip from Input (128)
  ├─ DoubleConv (384 → 384, residual)
  ├─ DoubleConv (384 → 128)
  └─ + Time Embedding (t)
   │
   ▼
Self-Attention (128, size=64)
   │
   ▼
Output Conv2d (128 → 3)
   │
   ▼
Predicted noise ──────────────────────────────────────────────────────────── </pre>

### 2. Diffusion 模型公式
* 前向加噪（Forward Process）
<pre>
x_t = sqrt(alpha_hat_t) * x_0 + sqrt(1 - alpha_hat_t) * epsilon
</pre>
* MSE
<pre>
E[|epsilon - epsilon(x_t,t)|^2]
</pre>

* 反向去躁
<pre>
x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise
</pre>

### 3. 訓練策略
* 採用 MSE Loss 作為訓練目標，預測每一步的加噪雜訊 
* 加入 **EMA（Exponential Moving Average）模型**，平滑主模型參數變動，提升生成穩定性與品質
* 每個 epoch 的最後 3 batch 用來估算 EMA loss，作為模型穩定性的指標

* Train Loss快速下降並於約 20 epoch 內趨於穩定。
* EMA Loss整體落在 0.01～0.03 間，但波動較大，顯示 EMA 模型的穩定性雖逐漸提升，但在某些 epoch 預測仍有偏差。


### 4. 參數
* Optimizer：AdamW
* Learning rate : CosineAnnealingLR
* batch_size   = 16
* epochs       = 160（最後測到 180）
* learning rate= 3e-4

* noise_steps  = 1000
* beta_start   = 1e-4
* beta_end     = 0.02

* EMA beta=0.998
### 5. 結果

雖然原本計畫訓練至 160~180 epoch，但實際觀察 FID 指標發現模型在**110 epoch**即已達最佳表現。可能因為模型可能在 110 之後出現輕微過擬合，生成圖片開始品質下降。

|epoch | FID |
|------|-----|
|**110** |**42.99(best)**|
120 |43.29
130 |43.54
140| 43.44
150 |43.96
155 |43.37

