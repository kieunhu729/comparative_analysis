import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Tạo thư mục chart
    os.makedirs("chart", exist_ok=True)
    
    # === 1. SƠ ĐỒ TRÒN (PIE CHART) VỀ PHÂN BỐ DỮ LIỆU ===
    categories = ['Simple', 'Complex', 'Idiom', 'Ambiguous']
    counts = [128, 35, 35, 2]
    colors = ['#4da6ff', '#ff9999', '#66b3ff', '#99ff99']
    explode = (0.05, 0, 0, 0) # Tách mảnh 'Simple' ra một nấc để nhấn mạnh
    
    plt.figure(figsize=(8, 6))
    plt.pie(counts, explode=explode, labels=categories, colors=colors, autopct='%1.1f%%', startangle=140, shadow=True)
    plt.title('Data Distribution by Complexity Category\n(Total: 200 Sentences)', fontsize=14, pad=20, fontweight='bold')
    plt.axis('equal')
    plt.savefig('chart/1_data_distribution_pie.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Data đã hard-code từ kết quả đánh giá (chính xác tuyệt đối)
    google_bleu = [10.99, 19.53, 33.44, 41.11]
    gemini_bleu = [23.46, 27.13, 33.44, 11.34]
    google_chrf = [31.24, 42.15, 42.17, 50.69]
    gemini_chrf = [43.21, 51.14, 42.17, 22.66]
    
    x = np.arange(len(categories))
    width = 0.35
    
    # === 2. SƠ ĐỒ CỘT GHÉP (BAR CHART) - ĐIỂM BLEU SCORES ===
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, google_bleu, width, label='Google Translate', color='#f4b400', edgecolor='grey')
    rects2 = ax.bar(x + width/2, gemini_bleu, width, label='Gemini 2.5 Flash', color='#4285f4', edgecolor='grey')
    
    ax.set_ylabel('BLEU Score', fontsize=12, fontweight='bold')
    ax.set_title('Translation Quality Comparison: BLEU Score by Category', fontsize=14, pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1)) # Dịch chú thích ra rìa
    
    # Hiển thị số liệu nhỏ xíu trên đỉnh các cột
    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')
    
    plt.ylim(0, max(max(google_bleu), max(gemini_bleu)) + 10) # Nới trần đồ thị
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('chart/2_bleu_score_comparison.png', dpi=300)
    plt.close()

    # === 3. SƠ ĐỒ CỘT GHÉP (BAR CHART) - ĐIỂM CHRF SCORES ===
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, google_chrf, width, label='Google Translate (NMT)', color='#db4437', edgecolor='grey')
    rects2 = ax.bar(x + width/2, gemini_chrf, width, label='Gemini (LLM)', color='#0f9d58', edgecolor='grey')
    
    ax.set_ylabel('chrF Score', fontsize=12, fontweight='bold')
    ax.set_title('Translation Morphological Quality Comparison: chrF Score', fontsize=14, pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.legend(loc='upper right')
    
    # Hiển thị số
    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')
    
    plt.ylim(0, 65)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('chart/3_chrf_score_comparison.png', dpi=300)
    plt.close()
    
    print("Đã vẽ xong và lưu 3 sơ đồ vào thư mục 'chart/'!")

if __name__ == "__main__":
    main()
