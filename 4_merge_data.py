import pandas as pd
import os

def main():
    print("Đang gộp dữ liệu từ Google và các batch của Gemini...")
    
    # 1. Đọc file dataset gốc
    df_main = pd.read_csv("data/evaluation_dataset_200.csv")
    
    # 2. Lấy dữ liệu Google Translate
    # Vì file dịch của Google lưu trong "data/translated_dataset_200.csv"
    google_file = "data/translated_dataset_200.csv"
    if os.path.exists(google_file):
        df_google = pd.read_csv(google_file)
        if "google_translate" in df_google.columns:
            df_main["google_translate"] = df_google["google_translate"]
            print("=> Đã nhúng 200 câu của Google Translate thành công.")
    else:
        print(f"-> Chưa thấy file kết quả Google ở '{google_file}', bỏ qua.")
        
    # 3. Gom nhặt 10 batch của Gemini (0->9)
    gemini_compiled = []
    # Tính số batch lấy trần dựa vào độ lớn file 
    total_batches = (len(df_main) + 19) // 20 
    
    for i in range(total_batches):
        batch_file = f"data/translate_gemini_batch_{i}.csv"
        if os.path.exists(batch_file):
            df_batch = pd.read_csv(batch_file)
            gemini_compiled.extend(df_batch["gemini_translate"].tolist())
        else:
            print(f"-> Cảnh báo: Thiếu {batch_file}, điền trống 20 câu.")
            gemini_compiled.extend([""] * 20)
            
    if any(gemini_compiled):
        # Đảm bảo độ dài array khớp với tổng số lượng dataset (tối đa cắt đi phần dư)
        df_main["gemini_translate"] = gemini_compiled[:len(df_main)]
        print("=> Đã nối các file chunk thành 1 cột hoàn chỉnh của Gemini Translate.")
        
    # 4. Ghi đè vào file file cuối cùng cho việc đánh giá
    output_file = "data/final_evaluation_dataset.csv"
    df_main.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nTuyệt vời! Tất cả đã được gộp gọn vào '{output_file}'. Bạn chạy 3_evaluate.py để tính điểm ngay bây giờ nhé!")

if __name__ == "__main__":
    main()
