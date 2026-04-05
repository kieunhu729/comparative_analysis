import pandas as pd
import time
import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("gemini_key") or os.getenv("GEMINI_KEY") or os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def classify_batch(texts_batch):
    # Dùng JSON constraint giống bài dịch để đảm bảo LLM phân loại đúng định dạng
    input_data = [{"id": i, "en": t} for i, t in enumerate(texts_batch)]
        
    prompt = """You are a linguistic expert analyzing English sentences. 
Task: Classify each sentence into EXACTLY ONE of the following 4 categories:
1. "simple" (Short, basic grammar, straightforward meaning, e.g., "I love you.")
2. "complex" (Long, compound/complex sentences, highly technical, or multiple clauses.)
3. "ambiguous" (Words with multiple meanings, missing context, or easily misunderstood without context.)
4. "idiom" (Contains idioms, phrasal verbs, slang, or metaphorical expressions requiring cultural context.)

Return EXACTLY a JSON array of objects with 'id' and 'category'. DO NOT wrap with markdown code blocks.
Example output format: [{"id": 0, "category": "complex"}, {"id": 1, "category": "idiom"}]
"""
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
            items = json.loads(response.text.strip())
            
            results = []
            for i in range(len(texts_batch)):
                found = next((item for item in items if item.get("id") == i), None)
                cat = found.get("category", "simple") if found else "simple"
                # chuẩn hóa phòng hờ
                cat = cat.lower().strip()
                if cat not in ["simple", "complex", "ambiguous", "idiom"]:
                    cat = "simple"
                results.append(cat)
                
            return results
        except Exception as e:
            print(f"Lỗi phân loại (thử lại {attempt+1}/{max_retries}): {e}")
            time.sleep(5)
            
    # Fallback
    return ["simple"] * len(texts_batch)

def main():
    file_path = "data/final_evaluation_dataset.csv"
    if not os.path.exists(file_path):
        print(f"Không tìm thấy {file_path}. Hãy gộp file trước.")
        return
        
    df = pd.read_csv(file_path)
    if "category" in df.columns:
        print("Cột 'category' đã tồn tại. Đang ghi đè...")
        
    english_texts = df['english'].tolist()
    
    # Chia batch 20
    batches = []
    batch_size = 20
    for i in range(0, len(english_texts), batch_size):
        batches.append(english_texts[i : i + batch_size])
        
    print(f"Bắt đầu phân loại {len(english_texts)} câu thành {len(batches)} batches...")
    all_categories = []
    
    for idx, batch_texts in enumerate(batches):
        print(f"Đang phân loại Batch {idx}...")
        cat_results = classify_batch(batch_texts)
        all_categories.extend(cat_results)
        
        # Limit 5 RPM
        if idx < len(batches) - 1:
            print("⏳ Nghỉ 15s để tránh Rate Limit...")
            time.sleep(15)
            
    # Đảm bảo bằng độ dài và gán vào Cột       
    df["category"] = all_categories[:len(df)]
    
    # Lưu lại file
    df.to_csv(file_path, index=False, encoding="utf-8")
    print(f"Hoàn tất thêm cột 'category'! Ghi đè vào {file_path}")

if __name__ == "__main__":
    main()
