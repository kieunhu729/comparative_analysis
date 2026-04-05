import pandas as pd
import os
from datasets import load_dataset

def main():
    print("Đang tải dataset tiếng Anh - tiếng Việt...")
    # Tải dataset opus100 (tập dữ liệu song ngữ rất phổ biến)
    dataset = load_dataset("opus100", "en-vi", split="train")
    
    # Trộn dữ liệu ngẫu nhiên và lấy ra 200 câu
    dataset = dataset.shuffle(seed=42)
    sample = dataset.select(range(200))
    
    data = []
    for item in sample:
        en_text = item['translation']['en']
        vi_text = item['translation']['vi']
        data.append({
            'english': en_text, 
            'human_reference_vi': vi_text # Bản dịch gốc bởi con người dùng để làm mốc so sánh
        })
        
    df = pd.DataFrame(data)
    
    # Tạo folder data nếu chưa có
    os.makedirs("data", exist_ok=True)
    
    # Lưu vào folder data
    df.to_csv("data/evaluation_dataset_200.csv", index=False, encoding='utf-8')
    print("Đã tạo xong file 'data/evaluation_dataset_200.csv' gồm 200 câu tiếng Anh và bản dịch mẫu.")
    
if __name__ == "__main__":
    main()
