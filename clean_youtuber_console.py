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
OUTPUT_CSV_FILE = "Yb6bjbWZaR8_final_console_result.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ID_TO_LABEL = {0: 'Clean (청정)', 1: 'Offensive (모욕)', 2: 'Hate (혐오)'}


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
        model = ElectraForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        print(f"✅ 모델 로드 완료! (Device: {device})")
        return tokenizer, model
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None, None


# ==============================================================================
# [2] 데이터 로드 및 전처리 (JSON 문자열 정리 포함)
# ==============================================================================
def clean_json_text(text):
    """
    '{"text": "실제 내용"}' 형태의 문자열에서 실제 내용만 추출합니다.
    """
    text = str(text).strip()
    # JSON 형식인 경우 파싱 시도
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

    # 1. 텍스트 파일로 읽어서 강제로 DataFrame 생성 (가장 안전)
    try:
        with open(filepath, 'r', encoding='utf-8-sig', errors='ignore') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        df = pd.DataFrame(lines, columns=['raw_text'])
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return None

    # 2. JSON 문자열 정리 (핵심 기능 추가)
    print("🧹 데이터 정제 중 (JSON 태그 제거)...")
    df['text'] = df['raw_text'].apply(clean_json_text)

    # 너무 짧은 글 제거
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

    print("🚀 AI 분석 시작...")
    preds_list = []
    probs_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing"):
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}

            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            preds_list.extend(preds.cpu().numpy())
            probs_list.extend([probs[j][p].item() for j, p in enumerate(preds)])

    df['Label_ID'] = preds_list
    df['Result'] = [ID_TO_LABEL[p] for p in preds_list]
    df['Confidence'] = [f"{p * 100:.1f}%" for p in probs_list]
    df['Status'] = ['BLOCK' if p != 0 else 'PASS' for p in preds_list]

    return df


# ==============================================================================
# [4] 결과 리포트 출력
# ==============================================================================
def print_report(df, text_col):
    total = len(df)
    toxic_df = df[df['Status'] == 'BLOCK']
    toxic_cnt = len(toxic_df)
    clean_cnt = total - toxic_cnt
    clean_score = (clean_cnt / total) * 100 if total > 0 else 0

    print("\n" + "=" * 60)
    print(f"📋 [분석 결과 리포트] : {INPUT_CSV_FILE}")
    print("=" * 60)
    print(f"🔹 총 댓글 수    : {total}개")
    print(f"🟢 청정 댓글     : {clean_cnt}개")
    print(f"🔴 악성 댓글     : {toxic_cnt}개")
    print(f"🛡️ 채널 청정 지수: {clean_score:.1f}점")
    print("-" * 60)

    print("\n🔢 [유형별 분포]")
    print(df['Result'].value_counts().to_string())

    if toxic_cnt > 0:
        print("\n🚨 [검출된 악성 댓글 샘플 (최대 5개)]")
        print("-" * 60)
        # 텍스트 길이제한을 두고 깔끔하게 출력
        for idx, row in toxic_df.head(5).iterrows():
            clean_content = row[text_col].replace("\n", " ")[:60]
            print(f"[{row['Result']}] {clean_content}...")
    else:
        print("\n✨ 악성 댓글이 발견되지 않았습니다.")

    # 저장 시에는 보기 좋게 필요한 컬럼만 저장
    save_df = df[['text', 'Result', 'Confidence', 'Status']]
    save_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print(f"💾 깔끔하게 정리된 결과가 '{OUTPUT_CSV_FILE}' 파일에 저장되었습니다.")
    print("=" * 60)


if __name__ == "__main__":
    tokenizer, model = load_model()
    if tokenizer:
        df, text_col = load_data_robust(INPUT_CSV_FILE)
        if df is not None and not df.empty:
            result_df = analyze(df, text_col, tokenizer, model)
            print_report(result_df, text_col)
