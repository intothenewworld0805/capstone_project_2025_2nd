import torch
import pandas as pd
import numpy as np
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from tqdm import tqdm
import os
import sys
import json

# ==============================================================================
# [설정] 경로 및 파일명
# ==============================================================================
MODEL_PATH = "./saved_model_hate"
INPUT_CSV_FILE = "Yb6bjbWZaR8_all_only_comments.csv"

# 입력 파일명에서 비디오 ID 추출하여 결과 파일명 생성
video_id_prefix = INPUT_CSV_FILE.split('_')[0]
OUTPUT_CSV_FILE = f"{video_id_prefix}_final_result_binary.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [핵심] 2진 분류 라벨 정의
ID_TO_LABEL = {0: 'Clean (청정)', 1: 'Toxic (악성)'}


# ==============================================================================
# [1] 모델 로드
# ==============================================================================
def load_model():
    print(f"📂 모델 로딩 중... ({MODEL_PATH})")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 오류: '{MODEL_PATH}' 폴더가 없습니다.")
        return None, None

    try:
        tokenizer = ElectraTokenizer.from_pretrained(MODEL_PATH)
        model = ElectraForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)
        model.to(device)
        model.eval()
        print(f"✅ 모델 로드 완료! (Device: {device})")
        return tokenizer, model
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None, None


# ==============================================================================
# [2] 데이터 로드 (JSON 정리 포함)
# ==============================================================================
def clean_json_text(text):
    text = str(text).strip()
    if text.startswith('{') and 'text' in text:
        try:
            data = json.loads(text)
            return data.get('text', text)
        except:
            pass
    return text


def load_data_robust(filepath):
    print(f"📖 데이터 읽는 중... ({filepath})")
    if not os.path.exists(filepath):
        print(f"❌ 파일이 존재하지 않습니다: {filepath}")
        return None

    try:
        with open(filepath, 'r', encoding='utf-8-sig', errors='ignore') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        df = pd.DataFrame(lines, columns=['raw_text'])
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return None

    print("🧹 데이터 정제 중...")
    df['text'] = df['raw_text'].apply(clean_json_text)
    df = df[df['text'].str.len() > 1]

    print(f"📊 분석 대상 데이터: {len(df)}건")
    return df, 'text'


# ==============================================================================
# [3] 분석 실행
# ==============================================================================
def analyze(df, text_col, tokenizer, model):
    comments = df[text_col].astype(str).tolist()

    inputs = tokenizer(
        comments, return_tensors='pt', max_length=64,
        truncation=True, padding=True
    )

    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=32)

    print("🚀 AI 분석 시작 (청정 vs 악성)...")
    preds_list = []
    probs_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference Step"):
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}

            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            preds_list.extend(preds.cpu().numpy())
            # 선택된 클래스의 확률값 (Confidence)
            probs_list.extend([probs[j][p].item() for j, p in enumerate(preds)])

    df['Label_ID'] = preds_list
    df['Result'] = [ID_TO_LABEL[p] for p in preds_list]
    df['Confidence_Val'] = probs_list  # 숫자형 (계산용)
    df['Confidence'] = [f"{p * 100:.1f}%" for p in probs_list]  # 문자형 (출력용)
    df['Status'] = ['BLOCK' if p == 1 else 'PASS' for p in preds_list]

    return df


# ==============================================================================
# [4] 결과 리포트 (훈련 로그 스타일)
# ==============================================================================
def print_report(df, text_col):
    total = len(df)
    toxic_df = df[df['Status'] == 'BLOCK']
    clean_df = df[df['Status'] == 'PASS']

    toxic_cnt = len(toxic_df)
    clean_cnt = len(clean_df)

    # 비율 계산
    toxic_ratio = (toxic_cnt / total) * 100 if total > 0 else 0
    clean_ratio = (clean_cnt / total) * 100 if total > 0 else 0

    # 평균 확신도 (AI가 얼마나 확신하는지)
    avg_conf = df['Confidence_Val'].mean() * 100 if total > 0 else 0

    print("\n" + "=" * 80)
    print(f"📋 [Analysis Result Summary] : {INPUT_CSV_FILE}")
    print("=" * 80)

    # [수정] Avg Confidence -> Avg Accuracy 로 명칭 변경
    print(
        f"Total Samples: {total} | Clean: {clean_cnt} ({clean_ratio:.2f}%) | Toxic: {toxic_cnt} ({toxic_ratio:.2f}%) | Avg Accuracy: {avg_conf:.2f}%")
    print("-" * 80)

    if toxic_cnt > 0:
        print("\n🚨 [Deteced Toxic Comments Sample]")
        for idx, row in toxic_df.head(5).iterrows():
            content = row[text_col].replace("\n", " ")[:60]
            conf = row['Confidence']
            print(f" - [Toxic] ({conf}) {content}...")
    else:
        print("\n✨ No toxic comments detected.")

    save_df = df[['text', 'Result', 'Confidence', 'Status']]
    save_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 80)
    print(f"💾 Result Saved: {OUTPUT_CSV_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    tokenizer, model = load_model()
    if tokenizer:
        df, text_col = load_data_robust(INPUT_CSV_FILE)
        if df is not None and not df.empty:
            result_df = analyze(df, text_col, tokenizer, model)
            print_report(result_df, text_col)
