import pandas as pd
import sacrebleu
import os

def main():
    file_path = "data/final_evaluation_dataset.csv"
    if not os.path.exists(file_path):
        print(f"Không tìm thấy {file_path}")
        return
        
    df = pd.read_csv(file_path)
    
    if "category" not in df.columns:
        print("Cột 'category' chưa tồn tại. Hãy chắc chắn rẳng script 5_classify_sentences.py đã chạy xong.")
        return
        
    print("===== PHÂN TÍCH CHẤT LƯỢNG DỊCH THEO TỪNG LOẠI CÂU =====")
    
    categories = df['category'].unique()
    
    for cat in categories:
        Cat_df = df[df['category'] == cat]
        print(f"\n--- NHÓM '{cat.upper()}' ({len(Cat_df)} câu) ---")
        
        # Human
        refs = [[ref] for ref in Cat_df['human_reference_vi'].tolist()]
        
        # Google
        if "google_translate" in Cat_df.columns:
            sys_gg = Cat_df["google_translate"].fillna("").tolist()
            bleu_gg = sacrebleu.corpus_bleu(sys_gg, refs).score
            chrf_gg = sacrebleu.corpus_chrf(sys_gg, refs).score
            print(f"  > Google Translate - BLEU: {bleu_gg:.2f} | chrF: {chrf_gg:.2f}")
            
        # Gemini
        if "gemini_translate" in Cat_df.columns:
            sys_gm = Cat_df["gemini_translate"].fillna("").tolist()
            bleu_gm = sacrebleu.corpus_bleu(sys_gm, refs).score
            chrf_gm = sacrebleu.corpus_chrf(sys_gm, refs).score
            print(f"  > Gemini Translate - BLEU: {bleu_gm:.2f} | chrF: {chrf_gm:.2f}")

    print("\n=> Phân tích này sẽ giúp bạn đưa thẳng vào báo cáo môn học phần 'Error Analysis'.")

if __name__ == "__main__":
    main()
