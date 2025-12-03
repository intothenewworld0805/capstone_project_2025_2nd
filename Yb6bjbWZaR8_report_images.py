import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
import matplotlib.font_manager as fm
import os

# 한글 폰트 설정 (Windows 기준)
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False


# 1. 학습 결과 그래프 생성 (Training Results)
def plot_training_history():
    epochs = [1, 2, 3]
    val_acc = [73.29, 76.90, 77.03]
    train_loss = [0.5404, 0.3668, 0.2237]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(epochs, train_loss, color=color, marker='o', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(epochs)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation Accuracy (%)', color=color)
    ax2.plot(epochs, val_acc, color=color, marker='s', label='Val Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('KoELECTRA Model Training History')
    fig.tight_layout()
    plt.grid(True)
    plt.savefig('report_training_graph.png')
    print("✅ 학습 그래프 저장 완료: report_training_graph.png")


# 2. 분석 결과 파이 차트 (Pie Chart)
def plot_result_pie(csv_file):
    if not os.path.exists(csv_file):
        print(f"❌ 파일 없음: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    # 컬럼명이 'Status' (BLOCK/PASS) 또는 'Result' (Clean/Toxic) 인지 확인
    target_col = 'Result' if 'Result' in df.columns else 'Analyzed_Label'

    counts = df[target_col].value_counts()

    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140,
            colors=['#ff9999', '#66b3ff'], explode=(0.05, 0))  # Toxic 강조
    plt.title(f'YouTube Comment Sentiment Analysis (N={len(df)})')
    plt.savefig('report_result_pie.png')
    print("✅ 결과 파이 차트 저장 완료: report_result_pie.png")


# 3. 악성 댓글 워드클라우드 (WordCloud)
def generate_wordcloud(csv_file):
    if not os.path.exists(csv_file):
        return

    df = pd.read_csv(csv_file)

    # 텍스트 컬럼 찾기
    text_col = 'text' if 'text' in df.columns else df.columns[0]

    # 악성 댓글만 추출 ('Toxic' 또는 'Hate'/'Offensive')
    # Result 컬럼에 'Toxic'이 있거나, Status가 'BLOCK'인 경우
    if 'Status' in df.columns:
        toxic_df = df[df['Status'] == 'BLOCK']
    else:
        # 혹시 모르니 전체 텍스트 사용 (필터링 로직에 따라 수정 가능)
        toxic_df = df

    text = " ".join(toxic_df[text_col].astype(str).tolist())

    # 워드클라우드 생성
    font_path = "C:/Windows/Fonts/malgun.ttf"
    wc = WordCloud(font_path=font_path, width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Toxic Comments Keyword Cloud')
    plt.savefig('report_wordcloud.png')
    print("✅ 워드클라우드 저장 완료: report_wordcloud.png")


if __name__ == "__main__":
    # 분석된 결과 파일명 입력
    INPUT_FILE = "Yb6bjbWZaR8_final_result_binary.csv"

    plot_training_history()
    plot_result_pie(INPUT_FILE)
    generate_wordcloud(INPUT_FILE)
