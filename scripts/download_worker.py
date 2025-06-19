import os
import json
import requests
from PIL import Image
from io import BytesIO
import time
import zipfile
import random
import hashlib
from tqdm import tqdm
import argparse
import cv2

# 初始化 DNN 模型
DNN_PROTO = "deploy.prototxt.txt"
DNN_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
dnn_net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)

def has_face(img_path, conf_threshold=0.5):
    img = cv2.imread(img_path)
    if img is None:
        return False
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), swapRB=False, crop=False)
    dnn_net.setInput(blob)
    detections = dnn_net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            return True
    return False


def is_clear_image(img_path, threshold=60.0):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, 0.0
    variance = cv2.Laplacian(img, cv2.CV_64F).var()
    return variance > threshold, variance

def is_face_and_clear(img_path):
    return has_face(img_path) and is_clear_image(img_path)

def get_filename_from_url(url):
    h = hashlib.md5(url.encode()).hexdigest()
    return f"{h}.jpg"

def download_image(url, save_path, headers):
    try:
        res = requests.get(url, timeout=5, headers=headers)
        if res.status_code == 200:
            img = Image.open(BytesIO(res.content)).convert("RGB")
            img.save(save_path, format="JPEG")
            return True
    except Exception:
        pass
    return False


def zip_and_cleanup(files, zip_name):
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in files:
            zipf.write(f, arcname=os.path.basename(f))
    for f in files:
        os.remove(f)

def process_month_file(jsonl_path, month_key):
    headers = {"User-Agent": "Mozilla/5.0"}
    temp_dir = f"temp_images_{month_key}"
    os.makedirs(temp_dir, exist_ok=True)

    image_count = 0
    batch_index = 1
    batch_images = []
    BATCH_SIZE = 500
    downloaded_urls = set()
    failed_log = f"failed_{month_key}.txt"
    filtered_log = f"filtered_out_{month_key}.txt"
    checkpoint_file = f"checkpoint_{month_key}.json"

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            ckpt = json.load(f)
            image_count = ckpt["image_count"]
            batch_index = ckpt["batch_index"]
            downloaded_urls = set(ckpt["downloaded_urls"])

    def save_ckpt():
        with open(checkpoint_file, "w") as f:
            json.dump({
                "image_count": image_count,
                "batch_index": batch_index,
                "downloaded_urls": list(downloaded_urls)
            }, f)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"處理月份 {month_key}"):
            article = json.loads(line)

            download_counter = 0  # 每 5 張 sleep 一次
            for url in article.get("images", []):
                if url in downloaded_urls:
                    continue
                filename = get_filename_from_url(url)
                save_path = os.path.join(temp_dir, filename)
                if os.path.exists(save_path):
                    downloaded_urls.add(url)
                    continue
                if download_image(url, save_path, headers):
                    face = has_face(save_path)
                    clear, sharpness = is_clear_image(save_path)
                    print(f" 原始網址：{url}")
                    print(f" 圖片：{save_path}")
                    print(f"   - 臉部偵測：{'有臉' if face else ' 無臉'}")
                    print(f"   - 清晰度值：{sharpness:.2f}（閾值 60） -> {'清晰' if clear else '模糊'}")
                    if face and clear:
                        batch_images.append(save_path)
                        downloaded_urls.add(url)
                        image_count += 1
                        download_counter += 1
                        print(f"已儲存清晰人臉：{save_path}")
                    else:
                        os.remove(save_path)
                        with open(filtered_log, "a") as flog:
                            flog.write(url + "\n")
                        print(f"過濾掉非人臉或模糊圖：{save_path}")
                else:
                    with open(failed_log, "a") as logf:
                        logf.write(url + "\n")
                if len(batch_images) >= BATCH_SIZE:
                    zip_and_cleanup(batch_images, f"images_{month_key}_part_{batch_index}.zip")
                    batch_index += 1
                    batch_images = []
                    save_ckpt()
                if download_counter > 0 and download_counter % 5 == 0:
                    print("已下載 5 張圖片，休息一下...")
                    time.sleep(random.uniform(0.5, 1.5))  # 每 5 張休息一次

    if batch_images:
        zip_and_cleanup(batch_images, f"images_{month_key}_part_{batch_index}.zip")
        save_ckpt()
    print(f" 月份 {month_key} 完成，共 {image_count} 張圖片")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Path to monthly jsonl")
    parser.add_argument("--month", required=True, help="Month key like '01', '02', etc.")
    args = parser.parse_args()
    process_month_file(args.jsonl, args.month)
