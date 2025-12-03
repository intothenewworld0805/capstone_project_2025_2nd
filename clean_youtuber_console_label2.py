import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from transformers import get_linear_schedule_with_warmup, logging
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from tqdm import tqdm
import os
import sys

# 0. 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 사용하는 장치: ", device)

# 1. 학습 시 경고 메세지 제거
logging.set_verbosity_error()

# ==============================================================================
# 2. 데이터 로드 및 라벨 병합 (핵심 수정)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_FILE = os.path.join(DATA_DIR, "train_beep.tsv")

if not os.path.exists(TRAIN_FILE):
    print(f"❌ 오류: 데이터 파일을 찾을 수 없습니다.")
    print("   'download_beep_data_check.py'를 먼저 실행해주세요.")
    sys.exit(1)

print(f"📖 데이터 읽는 중... {TRAIN_FILE}")
dataset = pd.read_csv(TRAIN_FILE, sep='\t').dropna(axis=0)

# ------------------------------------------------------------------------------
# [이전 코드 주석 처리] 3가지 분류 (청정/모욕/혐오)
# ------------------------------------------------------------------------------
# # 기존에는 offensive(1)와 hate(2)를 구분했습니다.
# label_map = {'none': 0, 'offensive': 1, 'hate': 2}
# dataset['label_id'] = dataset['hate'].map(label_map)
# ------------------------------------------------------------------------------

# [수정된 코드] 2가지 분류 (청정/악성)
# offensive(1)와 hate(2)를 모두 1(악성)로 통합하여 정확도를 높입니다.
label_map = {'none': 0, 'offensive': 1, 'hate': 1}
dataset['label_id'] = dataset['hate'].map(label_map)

text = list(dataset['comments'].values)
label = dataset['label_id'].values

print(f"\t * 학습 데이터 수: {len(text)}개")
# 라벨 분포 확인
print(f"\t * 라벨 분포 (0:청정, 1:악성): {dataset['label_id'].value_counts().to_dict()}")

# 3. 텍스트 토큰화
model_name = 'monologg/koelectra-base-v3-discriminator'
print(f"\n⚙️ 토크나이저 로드 ({model_name})...")
tokenizer = ElectraTokenizer.from_pretrained(model_name)

inputs = tokenizer(text, truncation=True, max_length=64, add_special_tokens=True,
                   padding="max_length")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# 4. 데이터 분리
train_ids, val_ids, train_labels, val_labels = train_test_split(input_ids, label, test_size=0.2, random_state=2025)
train_masks, val_masks, _, _ = train_test_split(attention_mask, label, test_size=0.2, random_state=2025)

# 5. Dataloader
batch_size = 32
train_data = TensorDataset(torch.tensor(train_ids), torch.tensor(train_masks), torch.tensor(train_labels))
train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)

val_data = TensorDataset(torch.tensor(val_ids), torch.tensor(val_masks), torch.tensor(val_labels))
val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size)

# ==============================================================================
# 6. 모델 설정 (num_labels 변경)
# ==============================================================================
print(f"🤖 모델 로드 중...")

# ------------------------------------------------------------------------------
# [이전 코드 주석 처리] 3가지 분류 모델
# ------------------------------------------------------------------------------
# model = ElectraForSequenceClassification.from_pretrained(model_name, num_labels=3)
# ------------------------------------------------------------------------------

# [수정된 코드] 2가지 분류 모델 (Binary Classification)
model = ElectraForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-08)
epochs = 3
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader) * epochs)

print("\n🔥 학습 시작! (청정 vs 악성 이진 분류)")
for e in range(epochs):
    # Training
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {e + 1}/{epochs} (Train)", leave=False)

    for batch in progress_bar:
        batch = tuple(t.to(device) for t in batch)
        model.zero_grad()
        outputs = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix({'loss': loss.item()})

    # Validation
    model.eval()
    val_preds, val_labels = [], []
    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            outputs = model(input_ids=batch[0], attention_mask=batch[1])
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        val_preds.extend(preds.cpu().numpy())
        val_labels.extend(batch[2].cpu().numpy())

    acc = np.sum(np.array(val_preds) == np.array(val_labels)) / len(val_preds)
    print(f"   Epoch {e + 1}: Avg Loss {total_loss / len(train_dataloader):.4f} | Val Accuracy {acc:.4f}")

# ==============================================================================
# 7. 모델 저장
# ==============================================================================
print("\n💾 모델 저장 중...")
save_path = os.path.join(BASE_DIR, "saved_model_hate")
if not os.path.exists(save_path):
    os.makedirs(save_path)

# [필수] 텐서 연속성 보장
for param in model.parameters():
    if not param.is_contiguous():
        param.data = param.data.contiguous()

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ 저장 완료! 경로: {save_path}")
