import pandas as pd
import time
import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Tự động load các biến môi trường từ file .env
load_dotenv()

api_key = os.getenv("gemini_key") or os.getenv("GEMINI_KEY") or os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def translate_batch(batch_idx, texts_batch):
    file_path = f"data/translate_gemini_batch_{batch_idx}.csv"
    
    # TINH NĂNG RESUME: Kiểm tra xem batch này đã được dịch thành công trước đó chưa
    if os.path.exists(file_path):
        try:
            df_existing = pd.read_csv(file_path)
            # Nếu tồn tại, kiểm tra dòng đầu tiên không bị trống
            if "gemini_translate" in df_existing.columns and len(df_existing) == len(texts_batch):
                first_item = str(df_existing["gemini_translate"].iloc[0]).strip()
                # DataFrame tạo từ null list thường có float 'nan'
                if first_item != "" and first_item.lower() != "nan":
                    print(f"[Batch {batch_idx}] Đã dịch xong từ lần chạy trước. Bỏ qua để tiết kiệm Quota.")
                    return "SKIPPED"
        except Exception:
            pass

    print(f"[Batch {batch_idx}] Bắt đầu dịch {len(texts_batch)} câu...")
    
    input_data = [{"id": i, "en": t} for i, t in enumerate(texts_batch)]
        
    prompt = "You are an expert English to Vietnamese translator. Translate the following English texts to Vietnamese.\n"
    prompt += "Return EXACTLY a JSON array of objects with 'id' and 'vi' keys where 'vi' is the translation. DO NOT wrap with markdown code blocks.\n"
    prompt += json.dumps(input_data, ensure_ascii=False)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json"
                ),
            )
            # Phân tích JSON list từ kết quả
            translated_items = json.loads(response.text.strip())
            
            # Gỡ ra thành danh sách đúng thứ tự id (chuẩn 100% không lộn xộn)
            results = []
            for i in range(len(texts_batch)):
                found = next((item for item in translated_items if item.get("id") == i), None)
                results.append(found.get("vi", "") if found else "")
            
            df_batch = pd.DataFrame({
                "english": texts_batch,
                "gemini_translate": results
            })
            df_batch.to_csv(file_path, index=False, encoding="utf-8")
            
            print(f"[Batch {batch_idx}] Dịch xong! Đã lưu an toàn vào {file_path}")
            return "SUCCESS"
            
        except Exception as e:
            err_msg = str(e)
            print(f"[Batch {batch_idx}] Lỗi (thử lại {attempt+1}/{max_retries}): {err_msg}")
            
            # Bắt đúng lỗi Quota Rate Limit 429
            if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                print("-> Bị kẹt Rate Limit 5 RPM. Đang ngủ 60 giây chờ hồi Quota...")
                time.sleep(60)
            else:
                time.sleep(5)
            
    print(f"[Batch {batch_idx}] Thất bại hoàn toàn sau {max_retries} lần thử lưu file trống...")
    df_batch = pd.DataFrame({"english": texts_batch, "gemini_translate": [""] * len(texts_batch)})
    df_batch.to_csv(file_path, index=False)
    return "FAILED"

def main():
    os.makedirs("data", exist_ok=True)
    df = pd.read_csv("data/evaluation_dataset_200.csv")
    english_texts = df['english'].tolist()
    
    batches = []
    batch_size = 20
    for i in range(0, len(english_texts), batch_size):
        batches.append((i // batch_size, english_texts[i : i + batch_size]))
        
    print(f"Tổng cộng có {len(batches)} batches. Sẽ chạy TUẦN TỰ để tránh vụ 5 RPM (Request Per Minute)...")
    
    for batch_idx, batch_texts in batches:
        status = translate_batch(batch_idx, batch_texts)
        if status == "SUCCESS":
            # API free chỉ cho 5 nháy mỗi phút -> tức là 1 nháy cách nhau 12 giây.
            # Ta nghỉ luôn 15 giây cho cực kì an toàn
            print("⏳ Đang nghỉ 15 giây để qua mặt giới hạn 5 API/phút của bản Free...")
            time.sleep(15)
        
    print("\nHoàn tất dịch tất cả các batch bằng Gemini! Hãy chạy python 4_merge_data.py để gộp lại.")

if __name__ == "__main__":
    main()
