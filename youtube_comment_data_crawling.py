# Youtube_API_KEY
# https://www.youtube.com/watch?v=Yb6bjbWZaR8

from googleapiclient.discovery import build
import json
from tqdm import tqdm
import os

# 🔐 본인의 API 키
API_KEY = 'YOUR_API_KEY'
VIDEO_ID = 'Yb6bjbWZaR8'
OUTPUT_FILE = '../../PycharmProjects/koelectra/data_crawling/Yb6bjbWZaR8_all_only_comments.csv'
CHECKPOINT_FILE = 'checkpoint.txt'

# YouTube API 클라이언트 생성
youtube = build('youtube', 'v3', developerKey=API_KEY)

def load_checkpoint():
    """
    checkpoint.txt 파일에서 마지막 nextPageToken을 불러옴
    """
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            token = f.read().strip()
            if token:
                print(f"🔁 체크포인트에서 재시작: {token}")
            return token if token else None
    return None

def save_checkpoint(token):
    """
    현재 nextPageToken을 checkpoint.txt에 저장
    """
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        if token:
            f.write(token)

def get_all_comments_with_checkpoint(video_id, output_file):
    """
    체크포인트 기능이 포함된 실시간 댓글 저장 (댓글만 저장)
    """
    next_page_token = load_checkpoint()
    page_count = 0
    comment_count = 0

    with open(output_file, 'a', encoding='utf-8') as f:  # 이어서 저장
        pbar = tqdm(desc="🔄 댓글 수집 중 (체크포인트 활성)", unit="페이지")
        while True:
            try:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    textFormat="plainText",
                    pageToken=next_page_token
                )
                response = request.execute()
                pbar.update(1)
                page_count += 1

                for item in response["items"]:
                    snippet = item["snippet"]["topLevelComment"]["snippet"]
                    comment_text = snippet.get("textDisplay", "")  # 댓글만 추출
                    json.dump({"text": comment_text}, f, ensure_ascii=False)  # 댓글만 저장
                    f.write('\n')
                    comment_count += 1

                next_page_token = response.get("nextPageToken")
                save_checkpoint(next_page_token)  # 진행상황 저장

                if not next_page_token:
                    break  # 더 이상 페이지 없음
            except Exception as e:
                print(f"⚠️ 오류 발생: {e}")
                break  # 오류 발생 시 종료

        pbar.close()

    print(f"✅ 수집 완료: 댓글 {comment_count}개, 페이지 {page_count}개")
    # 완료 시 checkpoint 삭제
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("🧹 체크포인트 파일 삭제 완료")

# 💡 실행
if __name__ == "__main__":
    print("🔍 유튜브 댓글 수집 시작 (체크포인트 지원)...")
    get_all_comments_with_checkpoint(VIDEO_ID, OUTPUT_FILE)
    print(f"💾 저장 완료: {OUTPUT_FILE}")
