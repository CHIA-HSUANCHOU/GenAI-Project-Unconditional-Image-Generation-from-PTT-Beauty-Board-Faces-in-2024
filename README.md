# Unconditional Image Generation from PTT Beauty Board Faces in 2024

---
Project: Unconditional Image Generation from PTT Beauty Board Faces in 2024

Author: CHOU CHIA-HSUAN

Date: 2025-05-27

Course: Generative AI

---

## 1. Code Structure
<pre> 
├── scripts/   #  Crawling, processing, training 
    ├── saveimagelinkmonth.py   # Extracts image links from PTT Beauty board and saves them by month
    ├── run_all_parallel_threaded.py  # Multi-thread controller: invokes downloader to process monthly tasks
    ├── download_worker.py # Downloads images and performs initial filtering for faces and clarity 
    ├── faceposition.py  # Image preprocessing (alignment, cropping, blur detection, classification)
    ├── diffusion.py # Generates images with diffusion model
    ├── Diffusion_submit.ipynb # Generates images with diffusion model
├── README.md                 

</pre> 

* Environment:

  * saveimagelinkmonth.py, run_all_parallel_threaded.py, and download_worker.py are executed locally.

  * faceposition.py, diffusion.py, and Diffusion_submit.ipynb are executed in Google Colab (images crawled locally are uploaded to Colab for training).

## 2. Crawl
`saveimagelinkmonth.py`
1. Identify all PTT Beauty board posts from January 1 to December 31, 2024, extract all image links (jpg/png/gif formats) from each post, and save them to image_links.jsonl.
2. Split the extracted links by month into image_links_01.jsonl, image_links_02.jsonl, ..., and save them into the monthly_jsonl/ folder.

`run_all_parallel_threaded.py, download_worker.py`
3. Use ThreadPoolExecutor(max_workers=4) to concurrently process up to 4 months of image downloads. The main controller script invokes download_worker.py to handle the JSONL data for each month.

4. After each image is downloaded, use OpenCV's DNN face detection model to retain only images containing a human face. Then compute the Laplacian variance to assess image clarity (threshold: 60). **Only images that contain a face and are clear will be saved**.

## 3. Preprocessing
`faceposition.py`

1. Image Quality Check
Each color image is converted to grayscale, and the Laplacian variance is computed to measure image sharpness. **Only images with variance above a threshold (60) are retained**.

2. Face Detection and Alignment
Faces are detected using InsightFace’s `app.get()` function, and only the largest detected face is kept. **Images are discarded if No face is detected, the detected face covers less than 1.5% of the image area**.

If detection succeeds, the angle between the eyes is calculated based on the left and right eye coordinates. The image is then rotated around the midpoint of the eyes using cv2.getRotationMatrix2D() to create an affine transformation matrix, followed by cv2.warpAffine() to **align the eyes horizontally**. To prevent cropping during rotation, the output size is set to 1.2× the original image size.

3. Second Detection and Cropping
After alignment, face detection is performed again. If successful, the face in the aligned image is used for cropping; otherwise, the fallback is the largest face in the original image.
**Cropping is centered on the face and expanded to cover the entire face area (including forehead and chin)**. A circular feather mask is created in the cropped region, with a radius equal to 0.55× the larger of face width or height. The center of the face is preserved, while the outer region is smoothly blended using Gaussian blur. The final blended image is resized to a fixed size (60×60).

4. Pose and Gender Classification
Based on the yaw angle of the face, the images are categorized as:

* frontal: |yaw| < 20°
* left: yaw < -20°
* right: yaw > 20°

Then, gender is predicted by the model:

* Gender = 1 → Male
* Gender = 0 → Female(If gender cannot be determined → Treated as Female)

5. Final Dataset Used
Originally, the plan was to train on frontal, left, and right faces separately. However, after counting, only ~4,000 images each were available for left and right profiles, which was insufficient for stable training.
To avoid instability due to data imbalance, **all poses and genders were merged into a single training set**. I manually filtered out side-face images with only one visible eye, incomplete facial features, and visibly blurred or unrecognizable faces.
The cleaned dataset is saved in the data/ directory, including both frontal and identifiable side faces, and is used to train the diffusion model.

## 4. Generation process
`diffusion.py`

### 1. Unet Structure

<pre> StrongerUNet Architecture ──────────────────────────────────────────────────────────── Input (x, t)
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

### 2. Diffusion 
* Forward Process
<pre>
x_t = sqrt(alpha_hat_t) * x_0 + sqrt(1 - alpha_hat_t) * epsilon
</pre>
* MSE
<pre>
E[|epsilon - epsilon(x_t,t)|^2]
</pre>

* Backward Process 
<pre>
x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise
</pre>

### 3. Training Strategy
* MSE Loss is used as the training objective, where the model learns to predict the added noise at each timestep.
* **EMA (Exponential Moving Average)** model is incorporated to smooth the parameter updates of the main model, improving generation stability and output quality.
* The last 3 batches of each epoch are used to estimate the EMA loss, serving as an indicator of the model's stability.
* Training loss drops rapidly and stabilizes within approximately 20 epochs.
* EMA loss generally falls between 0.01 and 0.03, though with noticeable fluctuations, indicating that while the EMA model becomes more stable over time, some epochs still show prediction inconsistencies.

### 4. Hyperparameters
* Optimizer：AdamW
* Learning rate : CosineAnnealingLR
* batch_size   = 16
* epochs       = 160（ (later extended to 180)）
* learning rate= 3e-4
* noise_steps  = 1000
* beta_start   = 1e-4
* beta_end     = 0.02
* EMA beta=0.998

### 5. Result
Although the original plan was to train for 160 to 180 epochs, observation of the FID metric revealed that the model achieved its best performance at around epoch **110**.
This suggests that the model may have begun to slightly **overfit after epoch 110**, leading to a decline in the quality of the generated images.


|epoch | FID |
|------|-----|
|**110** |**42.99(best)**|
120 |43.29
130 |43.54
140| 43.44
150 |43.96
155 |43.37

