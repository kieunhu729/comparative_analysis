import pandas as pd
import sacrebleu
import os

def main():
    file_path = "data/final_evaluation_dataset.csv"
    if not os.path.exists(file_path):
        file_path = "data/translated_dataset_200.csv"
        
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Không tìm thấy file '{file_path}'. Hãy chạy script dịch và gộp (4_merge_data) trước.")
        return

    # Điểm chuẩn (Human Reference)
    references = [[ref] for ref in df['human_reference_vi'].tolist()]
    
    if 'google_translate' in df.columns:
        google_preds = df['google_translate'].fillna("").tolist()
        print("================== ĐÁNH GIÁ GOOGLE TRANSLATE ==================")
        bleu_google = sacrebleu.corpus_bleu(google_preds, references)
        chrf_google = sacrebleu.corpus_chrf(google_preds, references)
        print(f"Điểm BLEU: {bleu_google.score:.2f}")
        print(f"Điểm chrF: {chrf_google.score:.2f}")

    if 'gemini_translate' in df.columns:
        gemini_preds = df['gemini_translate'].fillna("").tolist()
        print("\n================== ĐÁNH GIÁ GEMINI ============================")
        bleu_gemini = sacrebleu.corpus_bleu(gemini_preds, references)
        chrf_gemini = sacrebleu.corpus_chrf(gemini_preds, references)
        print(f"Điểm BLEU: {bleu_gemini.score:.2f}")
        print(f"Điểm chrF: {chrf_gemini.score:.2f}")
    
    if 'google_translate' not in df.columns and 'gemini_translate' not in df.columns:
         print("File CSV chưa có cột kết quả. Vui lòng chạy các file dịch trước!")

    print("\nLƯU Ý: Đây mới chỉ là đánh giá ĐỊNH LƯỢNG (Quantitative Metrics).\nTheo yêu cầu môn học, bạn bắt buộc phải đọc lại các file CSV, so sánh để tìm ra lỗi ngữ cảnh, văn phong, đại từ xưng hô, ngữ pháp... (Error Analysis & Linguistic Interpretation).")

if __name__ == "__main__":
    main()
