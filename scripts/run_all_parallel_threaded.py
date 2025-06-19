import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_worker(month):
    jsonl_path = f"monthly_jsonl/image_links_{month}.jsonl"
    cmd = f"python download_worker.py --jsonl {jsonl_path} --month {month}"
    print(f"[Month {month}] 任務開始")
    start = time.time()
    result = os.system(cmd)
    duration = time.time() - start
    print(f"[Month {month}] 任務結束（exit code={result}，耗時 {duration:.1f} 秒）")
    return month, result, duration

if __name__ == "__main__":
    months = [f"{i:02d}" for i in range(1, 13)]
    max_threads = 4

    print(f"使用 ThreadPoolExecutor（max_workers={max_threads}）")
    overall_start = time.time()

    results = []
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(run_worker, month) for month in months]
        for future in as_completed(futures):
            month, result, duration = future.result()
            results.append((month, result, duration))

    overall_duration = time.time() - overall_start
    print("\n所有月份處理結果：")
    for month, code, duration in sorted(results):
        status = "✅ 成功" if code == 0 else f"⚠️ 失敗（exit {code}）"
        print(f"  月份 {month}: {status}，耗時 {duration:.1f} 秒")

    print(f"\n全部任務完成，總耗時：{overall_duration:.1f} 秒")
