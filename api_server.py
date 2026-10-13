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
import time
from concurrent.futures import ThreadPoolExecutor
from warper import FaceWarper  # 匯入變形模組(表情)
from emotion import load_emotion_model, detect_emotion  # 匯入語音情緒偵測
from bgm import pick_bgm, build_mix_filter  # 匯入背景音樂 (依情緒挑歌與混音)

# ==========================================
# 1. 全域變數與模型預載 (Lifespan 模式)
# ==========================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
wav2lip_model = None
face_detector = None
emotion_model = None
user_states = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 啟動時執行 ---
    global wav2lip_model, face_detector, emotion_model
    print("\n[系統啟動] 正在載入 AI 模型")
    
    checkpoint_path = "checkpoints/wav2lip_gan.pth" 
    wav2lip_model = Wav2Lip()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    new_s = {k.replace('module.', ''): v for k, v in checkpoint["state_dict"].items()}
    wav2lip_model.load_state_dict(new_s)
    wav2lip_model = wav2lip_model.to(device).eval()

    face_detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, flip_input=False, device=device)

    # 情緒偵測模型常駐記憶體，每個請求只需推論
    emotion_model = load_emotion_model(device)
    print("模型載入完畢，伺服器已就緒！\n")
    yield
    # --- 關閉時執行 ---
    print("伺服器關閉中...")

app = FastAPI(lifespan=lifespan)

# ==========================================
# 2. LINE Bot 憑證與設定 (請確保與後台一致)
# ==========================================
# 金鑰放在專案根目錄的 .env (已被 .gitignore 忽略，不會推上 GitHub)，格式見 .env.example
def load_env_file(filename=".env"):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if not os.path.isfile(path):
        print(f"⚠️ 找不到 {filename}，請複製 .env.example 為 .env 並填入 LINE 金鑰")
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

load_env_file()
NGROK_URL = os.environ.get("NGROK_URL", "")                            # 你的 Ngrok 網址
CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")  # YOUR_CHANNEL_ACCESS_TOKEN
CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")              # YOUR_CHANNEL_SECRET

configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# 掛載結果資料夾，讓 LINE 可以存取影片網址
app.mount("/results", StaticFiles(directory="results"), name="results")

# ==========================================
# 3. 核心對嘴運算邏輯 (整合動態縮放與去模糊)
# ==========================================
# 表情 TPS 變形的平行執行緒數。
# 快取後最多只有 10 個不同的 (表情, ratio) 要算 (挑眉 7 + 眨眼 3)，
# 實測 10 執行緒 4.7x、8 執行緒 4.0x，再多沒有任務可分。
# OpenCV 的 warpImage 會釋放 GIL，所以用 thread 就能真的平行，不需要 multiprocessing。
WARP_THREADS = 10


def plan_expressions(total_frames, brow_dur=15, blink_dur=6, gap=4):
    """
    在整支影片的時間軸上排定「挑眉」與「眨眼」。

    保證：
      1. 兩種表情的區間彼此不重疊 (中間至少隔 gap 幀)，不會同時出現。
      2. 挑眉與眨眼在整段影片中各至少出現一次。

    回傳 [(kind, start_frame, duration), ...]；影片短到連一次都塞不下時回傳 []。
    """
    # 影片越長眨眼次數越多 (約每 3 秒眨一次)，但至少要有一次
    blink_count = max(1, total_frames // 75)

    # 需要的總長度 = 挑眉 + 所有眨眼 + 事件之間的間隔
    def span():
        return brow_dur + blink_dur * blink_count + gap * blink_count

    # 排不下時：先減少眨眼次數，還是不夠就縮短間隔並按比例壓縮動畫長度
    while blink_count > 1 and span() > total_frames:
        blink_count -= 1
    if span() > total_frames:
        gap = 1
        shrink = (total_frames - gap) / (brow_dur + blink_dur)
        brow_dur = max(2, int(brow_dur * shrink))
        blink_dur = max(2, int(blink_dur * shrink))
    if span() > total_frames:
        return []  # 音檔太短 (不到約 0.2 秒)，放棄表情

    events = [('brow', brow_dur)] + [('blink', blink_dur)] * blink_count
    random.shuffle(events)  # 隨機決定誰先誰後

    # 為每個事件抽一個「累進」的偏移量，讓表情隨機散布在整支影片上，
    # 偏移量排序後保證後面的事件不會往前壓到前一個事件。
    slack = total_frames - span()
    offsets = sorted(random.randint(0, slack) for _ in events)

    schedule = []
    cursor = 0
    for (kind, dur), off in zip(events, offsets):
        schedule.append((kind, cursor + off, dur))
        cursor += dur + gap
    return schedule


def process_wav2lip(face_path: str, audio_path: str, final_out_path: str, bgm: dict = None):
    # bgm: pick_bgm() 的結果；None = 不加背景音樂
    # 注意：Wav2Lip 的嘴型永遠只用 audio_path 的乾淨人聲計算，背景音樂只在最後編碼時混進音軌
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

    # --------------------------------------------------
    # 影片輸出：把每一幀的原始像素直接 pipe 給 ffmpeg，一次編成 h264 + 合併音軌
    # --------------------------------------------------
    # 舊做法是 cv2.VideoWriter 先壓成 DIVX avi 落地，再叫 ffmpeg 讀回來重壓成 mp4，
    # 等於編碼兩次 (多一次有損壓縮、多一次磁碟 I/O)，實測佔了純對嘴時間的近一半。
    # 改成單次編碼後畫質只會更好 (少一代 DIVX 損失)，也不再產生 _temp.avi。
    # 輸出格式維持與舊做法相同：h264 / yuv420p / aac；libx264 要求寬高為偶數，
    # 所以用 crop 切掉可能多出來的 1 像素 (舊做法 ffmpeg 內部也是這樣處理的)。
    # 幀的尺寸是「原始大圖的解析度」，因為我們最後要放大回去。
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{original_width}x{original_height}', '-r', str(fps), '-i', '-',   # 0: 畫面 (pipe)
        '-i', audio_path,                                                         # 1: 人聲
    ]
    if bgm is None:
        ffmpeg_cmd += ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']
    else:
        # 2: 背景音樂，-stream_loop -1 讓它無限循環，長度由 amix 以人聲為準截斷
        ffmpeg_cmd += [
            '-stream_loop', '-1', '-i', bgm["path"],
            '-filter_complex',
            '[0:v]crop=trunc(iw/2)*2:trunc(ih/2)*2[vout];' + build_mix_filter(bgm["gain_db"], len(wav) / 16000),
            '-map', '[vout]', '-map', '[aout]',
        ]
    ffmpeg_cmd += [
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        final_out_path,
    ]
    encoder = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # --------------------------------------------------
    # 新增：初始化表情產生器與時間軸
    # --------------------------------------------------
    total_frames = len(mel_chunks)

    warper = FaceWarper()
    landmarks_data = warper.get_landmarks(frame_small)

    # 把排程攤平成「幀 -> (表情種類, 進度)」對照表。
    # 一幀只會對應到一種表情，所以挑眉與眨眼不可能同時發生。
    expression_timeline = {}
    schedule = plan_expressions(total_frames) if landmarks_data else []
    for kind, start, dur in schedule:
        for f in range(start, start + dur):
            expression_timeline[f] = (kind, (f - start) / dur)

    if landmarks_data:
        print(f"[表情排程] 共 {total_frames} 幀 -> " +
              ", ".join(f"{k}@{s}~{s + d - 1}" for k, s, d in sorted(schedule, key=lambda e: e[1])))
    else:
        print("[表情排程] 偵測不到 FaceMesh 特徵點，本次不加入表情")

    # --------------------------------------------------
    # E. 開始批次生成與後處理 (Wav2Lip 推論 -> 放大 -> 去模糊)
    # --------------------------------------------------
    # 注意這裡多加了一個 frame_batch，用來記住每一幀對應的「變形後背景」
    # E. 開始批次生成與後處理 (Wav2Lip 推論 -> 放大 -> 去模糊)
    img_batch, mel_batch, frame_batch = [], [], []

    # --------------------------------------------------
    # 階段 1：先把所有需要的表情畫面一次算完 (快取 + 平行)
    # --------------------------------------------------
    # 來源圖 (frame_small) 與 landmarks 整支影片都固定，所以同一組 (表情, ratio)
    # 算出來的結果必然完全相同。而 sin(progress*pi) 是對稱的，挑眉 15 幀裡只有
    # 7 個不同的 ratio、眨眼 6 幀只有 3 個，去重後通常只剩 10 個要算。
    # 這 10 個彼此獨立，丟進 ThreadPool 平行做，比逐幀邊算邊用快 4~5 倍。
    #
    # 注意 1：key 一定要 round。sin(2pi/15) 和 sin(13pi/15) 數學上相等，但浮點數
    #         最後幾個位元會不一樣，不 round 的話去重幾乎不會命中。
    # 注意 2：快取裡的圖會被多個幀共用，絕對不能就地修改。下面貼回模型輸出時
    #         的 `f = frame_batch[j].copy()` 就是在保護這件事，那個 .copy() 不能拿掉。
    # 注意 3：warper.get_landmarks() 用到的 MediaPipe 不是 thread-safe，
    #         所以 landmarks 一定要在平行區之外先算好 (上面已經算了)。
    #         create_expression / create_blink 只讀常數清單，多執行緒共用同一個 warper 是安全的。
    def warp_key(frame_idx):
        """回傳 (快取 key, 精確 ratio)；ratio 為 0 (每段表情的第一幀) 回傳 (None, 0) 表示直接用原圖"""
        kind, progress = expression_timeline[frame_idx]
        amount = math.sin(progress * math.pi)  # 0 -> 1 -> 0
        return ((kind, round(amount, 6)), amount) if amount > 0 else (None, 0.0)

    # key 用四捨五入後的值來去重，但實際計算要用「第一次遇到的精確值」。
    # 若直接拿四捨五入的值去算，4e-7 的誤差會讓少數控制點的 float32 座標翻一個 ulp，
    # 結果就不再和逐幀計算的版本位元級相同 (實測會有幾幀差 1~18)。
    needed = {}
    for frame_idx in sorted(expression_timeline):
        key, exact_amount = warp_key(frame_idx)
        if key is not None:
            needed.setdefault(key, exact_amount)

    def make_warp(item):
        key, amount = item
        kind = key[0]
        landmarks, h, w = landmarks_data
        if kind == 'brow':
            return key, warper.create_expression(frame_small.copy(), landmarks, h, w, amount * 0.02)
        return key, warper.create_blink(frame_small.copy(), landmarks, h, w, amount)

    warp_cache = {}
    if needed:
        t_warp = time.perf_counter()
        n_threads = min(len(needed), WARP_THREADS)
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            warp_cache = dict(pool.map(make_warp, needed.items()))
        print(f"[表情變形] {len(expression_timeline)} 幀表情 -> 去重後 {len(needed)} 次 TPS，"
              f"{n_threads} 執行緒平行，耗時 {time.perf_counter() - t_warp:.2f} s")

    # --------------------------------------------------
    # 階段 2：逐幀查表，組 batch 餵模型
    # --------------------------------------------------
    for i, m in enumerate(mel_chunks):
        # 沒表情的幀直接引用原圖 (不 copy)，下游會自己 copy 再寫入
        current_frame_img = frame_small

        if i in expression_timeline:
            key, _ = warp_key(i)
            if key is not None:
                current_frame_img = warp_cache[key]

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

                encoder.stdin.write(f_sharpened.tobytes())
                
            img_batch, mel_batch, frame_batch = [], [], []
            
    # --------------------------------------------------
    # F. 關閉 pipe，等 ffmpeg 把最後的幀編完並寫出 mp4
    # --------------------------------------------------
    # 所有幀都送完了，關掉 stdin 讓 ffmpeg 知道沒有更多輸入；communicate 會等它結束並收 stderr
    _, ffmpeg_err = encoder.communicate()
    if encoder.returncode != 0:
        return False, f"ffmpeg 編碼失敗: {ffmpeg_err.decode('utf-8', errors='replace').strip()[:300]}"
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

        # 1.5 語音情緒偵測 + 挑背景音樂 (失敗就不加配樂，不影響影片生成)
        bgm = None
        try:
            t0 = time.time()
            wav = audio.load_wav(wav_path, 16000)
            emotion = detect_emotion(emotion_model, wav)
            scores = ", ".join(f"{k}={v:.2f}" for k, v in sorted(emotion["scores"].items(), key=lambda kv: -kv[1])[:3])
            print(f"[情緒] {emotion['top']} | 分組={emotion['group']} | {scores} | {time.time() - t0:.2f}s")

            bgm = pick_bgm(emotion, wav)
            if bgm:
                print(f"[配樂] {os.path.basename(bgm['path'])} | 增益 {bgm['gain_db']:+.1f}dB")
            else:
                print("[配樂] 不加背景音樂")
        except Exception as e:
            print(f"[情緒/配樂] 失敗，這次不加背景音樂: {e}")

        # 2. 生成影片
        final_name = f"result_{user_id}.mp4"
        final_path = f"results/{final_name}"
        success, msg = process_wav2lip(face_path, wav_path, final_path, bgm=bgm)

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