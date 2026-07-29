from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse   
from contextlib import asynccontextmanager
import uvicorn
import torch
import cv2, os, subprocess,sys
import numpy as np
import audio
import face_detection
from models import Wav2Lip
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import FlexMessage, FlexContainer
from linebot.v3.messaging import (
    Configuration, 
    ApiClient, 
    MessagingApi, 
    MessagingApiBlob,
    ReplyMessageRequest, 
    PushMessageRequest,
    TextMessage, 
    VideoMessage
)
from linebot.v3.webhooks import MessageEvent, ImageMessageContent, AudioMessageContent, TextMessageContent

import math
import random
from warper import FaceWarper  # 匯入我們剛寫好的變形模組

# ==========================================
# 1. 全域變數與模型預載 (Lifespan 模式)
# ==========================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
wav2lip_model = None
face_detector = None
user_states = {} 

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 啟動時執行 ---
    global wav2lip_model, face_detector
    print("\n[系統啟動] 正在載入 AI 模型")
    
    checkpoint_path = "checkpoints/wav2lip_gan.pth" 
    wav2lip_model = Wav2Lip()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    new_s = {k.replace('module.', ''): v for k, v in checkpoint["state_dict"].items()}
    wav2lip_model.load_state_dict(new_s)
    wav2lip_model = wav2lip_model.to(device).eval()

    face_detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, flip_input=False, device=device)
    print("模型載入完畢，伺服器已就緒！\n")
    yield
    # --- 關閉時執行 ---
    print("伺服器關閉中...")

app = FastAPI(lifespan=lifespan)

# ==========================================
# 2. LINE Bot 憑證與設定 (請確保與後台一致)
# ==========================================
NGROK_URL = "你的 Ngrok 網址" # 你的 Ngrok 網址
CHANNEL_ACCESS_TOKEN = 'YOUR_CHANNEL_ACCESS_TOKEN'
CHANNEL_SECRET = 'YOUR_CHANNEL_SECRET'

configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# 掛載結果資料夾，讓 LINE 可以存取影片網址
app.mount("/results", StaticFiles(directory="results"), name="results")

# ==========================================
# 3. 核心對嘴運算邏輯 (整合動態縮放與去模糊)
# ==========================================
def process_wav2lip(face_path: str, audio_path: str, final_out_path: str):
    out_temp = final_out_path.replace('.mp4', '_temp.avi')
    
    if not os.path.isfile(face_path) or not os.path.isfile(audio_path):
        return False, "找不到原始檔案"

    # --------------------------------------------------
    # A. 處理音訊 (維持不變)
    # --------------------------------------------------
    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)
    fps = 25.0
    mel_chunks = []
    i = 0
    while 1:
        start_idx = int(i * (80./fps))
        if start_idx + 16 > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - 16:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + 16])
        i += 1

    # --------------------------------------------------
    # B. 讀取原圖與第一次人臉偵測 (取得原始座標)
    # --------------------------------------------------
    frame = cv2.imread(face_path)
    original_height, original_width = frame.shape[:2]

    rects = face_detector.get_detections_for_batch(np.array([frame]))
    rect = rects[0]
    if rect is None: return False, "影像中找不到人臉"

    orig_y1, orig_y2 = max(0, rect[1]), min(original_height, rect[3]+20)
    orig_x1, orig_x2 = max(0, rect[0]), min(original_width, rect[2])

    face_width = orig_x2 - orig_x1
    face_height = orig_y2 - orig_y1

    # --------------------------------------------------
    # C. 動態計算縮放比例 (確保人臉最小不低於 256)
    # --------------------------------------------------
    MIN_FACE_SIZE = 256
    current_min_face_side = min(face_width, face_height)
    scale_factor = MIN_FACE_SIZE / current_min_face_side
    scale_factor = min(1.0, scale_factor) # 安全限制：最多只縮小，不主動放大原圖

    # --------------------------------------------------
    # D. 執行縮放與座標映射
    # --------------------------------------------------
    if scale_factor < 1.0:
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        # 整圖降解析度 (運算用的小圖)
        frame_small = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # 映射新的人臉座標
        y1, y2 = int(orig_y1 * scale_factor), int(orig_y2 * scale_factor)
        x1, x2 = int(orig_x1 * scale_factor), int(orig_x2 * scale_factor)
    else:
        frame_small = frame.copy()
        y1, y2, x1, x2 = int(orig_y1), int(orig_y2), int(orig_x1), int(orig_x2)

    # 裁切要餵給 Wav2Lip 的臉部區域 (Wav2Lip 依然需要 96x96)
    face_img = cv2.resize(frame_small[y1:y2, x1:x2], (96, 96))

    # 注意：VideoWriter 必須設定為「原始大圖的解析度」，因為我們最後要放大回去
    out = cv2.VideoWriter(out_temp, cv2.VideoWriter_fourcc(*'DIVX'), fps, (original_width, original_height))

    # --------------------------------------------------
    # 新增：初始化表情產生器與時間軸
    # --------------------------------------------------
    total_frames = len(mel_chunks)
    expression_duration = 15 # 挑眉動畫持續 15 幀
    
    # 隨機決定挑眉發生的起始時間
    if total_frames > expression_duration:
        start_expr_frame = random.randint(0, total_frames - expression_duration)
    else:
        start_expr_frame = -1

    warper = FaceWarper()
    landmarks_data = warper.get_landmarks(frame_small)

    # --------------------------------------------------
    # E. 開始批次生成與後處理 (Wav2Lip 推論 -> 放大 -> 去模糊)
    # --------------------------------------------------
    # 注意這裡多加了一個 frame_batch，用來記住每一幀對應的「變形後背景」
    # E. 開始批次生成與後處理 (Wav2Lip 推論 -> 放大 -> 去模糊)
    img_batch, mel_batch, frame_batch = [], [], []
    
    for i, m in enumerate(mel_chunks):
        current_frame_img = frame_small.copy()
        
        # 如果這一幀剛好落在要挑眉的區間內，就執行變形
        if landmarks_data and (start_expr_frame <= i < start_expr_frame + expression_duration):
            landmarks, h, w = landmarks_data
            progress = (i - start_expr_frame) / expression_duration
            lift_ratio = math.sin(progress * math.pi) * 0.02
            current_frame_img = warper.create_expression(current_frame_img, landmarks, h, w, lift_ratio)
        
        current_face_crop = cv2.resize(current_frame_img[y1:y2, x1:x2], (96, 96))
        
        img_batch.append(current_face_crop)
        mel_batch.append(m)
        frame_batch.append(current_frame_img)
        
        if len(img_batch) >= 128 or i == len(mel_chunks) - 1:
            img_batch_np = np.asarray(img_batch)
            img_masked = img_batch_np.copy()
            img_masked[:, 96//2:] = 0
            img_batch_np = np.concatenate((img_masked, img_batch_np), axis=3) / 255.
            mel_batch_np = np.reshape(np.asarray(mel_batch), [len(mel_batch), 80, 16, 1])
            
            img_t = torch.FloatTensor(np.transpose(img_batch_np, (0, 3, 1, 2))).to(device)
            mel_t = torch.FloatTensor(np.transpose(mel_batch_np, (0, 3, 1, 2))).to(device)
            
            with torch.no_grad():
                pred = wav2lip_model(mel_t, img_t).cpu().numpy().transpose(0, 2, 3, 1) * 255.
                
            for j, p in enumerate(pred):
                f = frame_batch[j].copy() 
                
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p
                
                if scale_factor < 1.0:
                    f_upscaled = cv2.resize(f, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
                else:
                    f_upscaled = f

                gaussian = cv2.GaussianBlur(f_upscaled, (0, 0), 2.0)
                f_sharpened = cv2.addWeighted(f_upscaled, 1.5, gaussian, -0.5, 0)

                out.write(f_sharpened)
                
            img_batch, mel_batch, frame_batch = [], [], []
            
    # 確保 out.release() 只有一個，並且退到與 for 迴圈同一層級
    out.release()
    
    # --------------------------------------------------
    # F. 使用 FFmpeg 合併音軌 (維持不變)
    # --------------------------------------------------
    subprocess.call(f'ffmpeg -y -i {audio_path} -i {out_temp} -strict -2 -q:v 1 {final_out_path} -loglevel error', shell=True)
    if os.path.exists(out_temp): os.remove(out_temp)
    return True, final_out_path
# ==========================================
# 4. 提供 LIFF 播放器網頁
# ==========================================
@app.get("/player")
def serve_player_page():
    return FileResponse("static/player.html")

def get_source_id(event):
    """判斷訊息來源是個人、群組還是聊天室，並回傳對應的 ID"""
    if event.source.type == 'group':
        return event.source.group_id
    elif event.source.type == 'room':
        return event.source.room_id
    else:
        return event.source.user_id

# ==========================================
# 5. LINE Webhook 處理
# ==========================================
@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get('X-Line-Signature')
    body = await request.body()
    try:
        handler.handle(body.decode('utf-8'), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400)
    return 'OK'

#  新增：處理文字訊息 (處理選單與talking face開關)
@handler.add(MessageEvent, message=TextMessageContent)
def handle_text(event):
    text = event.message.text.strip()
    user_id = event.source.user_id
    source_id = get_source_id(event) 

    # ==========================================
    # 情境 1：使用者發送了 "@Tang_Project_group"
    # ==========================================
    if text == "@Tang_Project_group":
        # 將該使用者標記為「等待接收照片」的狀態
        user_states[user_id] = {
            "state": "WAIT_IMAGE",
            "source_id": source_id
        }
        reply_text(event.reply_token, " 已開啟talking face模式！\n請在此聊天室上傳一張包含人臉的照片。")
        
    # 如果是一般聊天內容，機器人就不做任何事

@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image(event):
    user_id = event.source.user_id
    
    #  狀態檢查：如果該使用者不在狀態紀錄中，或是狀態不是「等待照片」，就直接忽略這張圖
    if user_id not in user_states or user_states[user_id].get("state") != "WAIT_IMAGE":
        return 

    with ApiClient(configuration) as api_client:
        line_api_blob = MessagingApiBlob(api_client)
        content = line_api_blob.get_message_content(event.message.id)
        path = f"inputs/face_{user_id}.jpg"
        with open(path, "wb") as f: f.write(content)
            
    # 更新狀態為「等待音檔」，並存入照片路徑
    user_states[user_id]["state"] = "WAIT_AUDIO"
    user_states[user_id]["face_path"] = path
    
    reply_text(event.reply_token, "照片已就緒！請接著傳送語音訊息開始生成影片。")

@handler.add(MessageEvent, message=AudioMessageContent)
def handle_audio(event):
    user_id = event.source.user_id
    
    # 狀態檢查：如果該使用者不是處於「等待音檔」狀態，就直接忽略這段錄音
    if user_id not in user_states or user_states[user_id].get("state") != "WAIT_AUDIO":
        return

    # 取出這個任務要回傳到哪個群組/個人，以及他剛剛傳的照片
    source_id = user_states[user_id]["source_id"] 
    face_path = user_states[user_id]["face_path"]      #照片路徑

    with ApiClient(configuration) as api_client:
        line_api = MessagingApi(api_client)
        line_api_blob = MessagingApiBlob(api_client)
        
        # 1. 下載並轉檔 
        audio_content = line_api_blob.get_message_content(event.message.id)
        m4a_path = f"inputs/audio_{user_id}.m4a"
        wav_path = f"inputs/audio_{user_id}.wav"
        with open(m4a_path, "wb") as f: f.write(audio_content)
        subprocess.call(f'ffmpeg -y -i {m4a_path} -ar 16000 {wav_path} -loglevel error', shell=True)

        # 2. 生成影片 
        final_name = f"result_{user_id}.mp4"
        final_path = f"results/{final_name}"
        success, msg = process_wav2lip(face_path, wav_path, final_path)

        # 3. 推送結果
        if success:
            liff_id = "2009716676-IRSKacfn"
            liff_url = f"https://liff.line.me/{liff_id}?video={final_name}"
            reply_msg = f" 專屬影片已完成！\n點擊下方連結觀看，播完會自動回到聊天室喔\n{liff_url}"
            
            line_api.push_message(PushMessageRequest(
                to=source_id, 
                messages=[TextMessage(text=reply_msg)]
            ))
        else:
            line_api.push_message(PushMessageRequest(
                to=source_id, 
                messages=[TextMessage(text=f"❌ 錯誤: {msg}")]
            ))
            
        # 任務完成！刪除該使用者的狀態，讓他回到一般聊天模式
        del user_states[user_id]

def reply_text(token, text):
    with ApiClient(configuration) as api_client:
        MessagingApi(api_client).reply_message(ReplyMessageRequest(reply_token=token, messages=[TextMessage(text=text)]))

if __name__ == "__main__":
    os.makedirs("inputs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000)