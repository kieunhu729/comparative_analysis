import pandas as pd
import time
import os
from deep_translator import GoogleTranslator

def batch_translate_google(texts):
    translations = []
    print("Bắt đầu dịch bằng Google Translate (deep-translator)...")
    translator = GoogleTranslator(source='en', target='vi')
    for idx, text in enumerate(texts):
        try:
            res = translator.translate(text)
            translations.append(res)
        except Exception as e:
            print(f"Lỗi Google ở dòng {idx}: {e}")
            translations.append("")
        time.sleep(0.1) # Tránh bị block
    return translations

def main():
    target_file = "data/translated_dataset_200.csv"
    # Kiểm tra xem file dịch đã tồn tại chưa để nối thêm cột vào
    if os.path.exists(target_file):
        df = pd.read_csv(target_file)
    else:
        df = pd.read_csv("data/evaluation_dataset_200.csv")
        
    english_texts = df['english'].tolist()
    
    # Bắt đầu dịch
    df['google_translate'] = batch_translate_google(english_texts)
    
    # Ghi đè lại vào file kết quả
    df.to_csv(target_file, index=False, encoding='utf-8')
    print(f"Hoàn tất dịch Google và lưu vào {target_file}.")

if __name__ == "__main__":
    main()
