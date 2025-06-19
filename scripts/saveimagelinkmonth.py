import requests
import time
import json
import random
import re
from bs4 import BeautifulSoup
from collections import defaultdict
from datetime import datetime
import os

PTT_URL = 'https://www.ptt.cc'
BOARD = 'Beauty'
session = requests.Session()
session.cookies.set('over18', '1')

articles = []
image_link_output = "image_links.jsonl"
img_pattern = re.compile(r'https?://[^\s]+\.(?:jpg|jpeg|png|gif)', re.IGNORECASE)

def is_valid_title(title):
    if not title:
        return False
    title = title.strip()
    if title == "":
        return False
    if '[公告]' in title or 'Fw:[公告]' in title:
        return False
    return True

def detect_date_on_page(index):
    url = f"{PTT_URL}/bbs/{BOARD}/index{index}.html"
    res = session.get(url, timeout=5)
    if res.status_code != 200:
        print(f"無法讀取 index{index}")
        return []

    soup = BeautifulSoup(res.text, 'html.parser')
    entries = soup.select('div.r-ent')

    dates = []
    for entry in entries:
        date_tag = entry.select_one('div.date')
        if not date_tag:
            continue
        try:
            raw_date = date_tag.text.strip()
            month, day = map(int, raw_date.split('/'))
            dates.append(datetime(2024, month, day))
        except:
            continue
    return dates


def find_start_index():
    idx = 3647  #3647
    last_valid = None
    while idx > 3000:
        dates = detect_date_on_page(idx)
        if dates is None:
            print(f"index{idx} 無法取得資料，跳過")
            idx -= 1
            continue

        if any(d == datetime(2024, 1, 1) for d in dates):
            last_valid = idx  
        elif last_valid is not None:
            break

        idx -= 1
        time.sleep(random.uniform(0.3, 0.8))

    return last_valid

def find_end_index():
    idx = 3916  #3916 
    while idx < 4000:
        dates = detect_date_on_page(idx)
        if any(d == datetime(2024, 12, 31) for d in dates):
            return idx
        idx -= 1
        time.sleep(random.uniform(0.3, 0.8))
    return None

def extract_valid_text(text):
    lines = text.split("\n")
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if start_idx is None and line.startswith("作者"):
            start_idx = i
        if line.startswith("※ 發信站"):
            end_idx = i
            break
    if start_idx is not None and end_idx is not None:
        return "\n".join(lines[start_idx:end_idx])
    else:
        return None
    
def extract_image_links(article_url):
    try:
        res = session.get(article_url, timeout=5)
        if res.status_code != 200:
            print(f"無法開啟文章：{res.status_code} {article_url}")
            return []
        soup = BeautifulSoup(res.text, 'html.parser')
        main_content = soup.select_one("#main-content")
        if main_content:
            text = main_content.get_text()
            valid_text = extract_valid_text(text)
            urls = img_pattern.findall(valid_text )
            print(f" 找到圖片連結數量：{len(urls)}")
            return urls
        return []
    except Exception as e:
        print(f"錯誤讀取文章：{article_url} -> {e}")
        return []

def parse_index_page(page_url, is_first_page=False, is_last_page=False):
    res = session.get(page_url, timeout=5)
    if res.status_code != 200:
        print(f"無法取得頁面: {page_url}")
        return None
    soup = BeautifulSoup(res.text, 'html.parser')
    entries = soup.select('div.r-ent')

    count_since_last_sleep = 0
    for entry in entries:
        title_tag = entry.select_one('div.title > a')
        if not title_tag:
            continue
        title = title_tag.text.strip()
        if not is_valid_title(title):
            continue
        url = title_tag['href']
        full_url = PTT_URL + url
        date_tag = entry.select_one('div.date')
        if not date_tag:
            continue
        try:
            month, day = map(int, date_tag.text.strip().split('/'))
            if is_first_page and month != 1:
                continue
            if is_last_page and month == 1 and day == 1:
                break
            date_str = f"{month:02d}{day:02d}"
        except:
            continue

        print(f"\n開始處理文章連結：{full_url}")
        images = extract_image_links(full_url)

        if images:
            post = {
                "date": date_str,
                "title": title,
                "url": full_url,
                "images": images
            }
            articles.append(post)
                
        count_since_last_sleep += 1    
        if count_since_last_sleep == 10:
            print("休息一下...")
            time.sleep(random.uniform(0.5, 1.5))  # 每 10 篇暫停一下
            count_since_last_sleep = 0

    return soup

def main():
    start_time = time.time()  # 開始計時
    start_idx = find_start_index()
    end_idx = find_end_index()
    #start_idx = 3647
    #end_idx = 3648
    for i in range(start_idx, end_idx + 1):
        page_url = f"{PTT_URL}/bbs/{BOARD}/index{i}.html"
        is_first = (i == start_idx)
        is_last = (i == end_idx)
        soup = parse_index_page(page_url, is_first_page=is_first, is_last_page=is_last)
        if soup is None:
            continue
        time.sleep(random.uniform(0.5, 1.5))

    with open(image_link_output, "w", encoding="utf-8") as f:
        for a in articles:
            json.dump(a, f, ensure_ascii=False)
            f.write("\n")
    print(f"\n已儲存 {len(articles)} 筆文章連結至 {image_link_output}")
    total_images = sum(len(a["images"]) for a in articles)
    print(f"共找到圖片連結總數：{total_images} 張")
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"總共花費時間：{elapsed:.2f} 秒")
    

INPUT_JSONL = "image_links.jsonl"
OUTPUT_DIR = "monthly_jsonl"
os.makedirs(OUTPUT_DIR, exist_ok=True)

monthly_data = defaultdict(list)
monthly_image_counts = defaultdict(int)

with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        date_str = record.get("date")
        if not date_str or len(date_str) < 2:
            continue
        try:
            month = int(date_str[:2])
            if 1 <= month <= 12:
                month_key = f"{month:02d}"
                monthly_data[month_key].append(record)
                monthly_image_counts[month_key] += len(record.get("images", []))
        except ValueError:
            print(f"無法解析月份：{date_str}")

# 寫入分月的檔案
for month_key, records in monthly_data.items():
    out_path = os.path.join(OUTPUT_DIR, f"image_links_{month_key}.jsonl")
    with open(out_path, "w", encoding="utf-8") as out_file:
        for record in records:
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"已成功拆成 {len(monthly_data)} 個月份檔案，儲存在 {OUTPUT_DIR}/\n")

# 顯示每月圖片總數
print("每月圖片總數統計：")
for month in sorted(monthly_image_counts.keys()):
    print(f"  月份 {month}: {monthly_image_counts[month]} 張圖片")

if __name__ == "__main__":
    main()