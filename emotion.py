"""
語音情緒偵測 (emotion2vec_plus_large)

用法：
    model = load_emotion_model(device)          # 伺服器啟動時載入一次，常駐記憶體
    result = detect_emotion(model, wav_16k)     # 每個請求只做推論

模型放在專案內的 checkpoints/emotion2vec_plus_large/ (已被 .gitignore 忽略)。
資料夾不存在時 (例如換電腦部署) 會自動從 HuggingFace 下載約 1.9GB 到同一個位置。
(不走 ModelScope：modelscope 新版用了 Python 3.9+ 語法，這個 venv 是 3.8 會直接 import 失敗)
"""
import contextlib
import io
import logging
import os

import numpy as np

EMOTION_MODEL_ID = "emotion2vec/emotion2vec_plus_large"
# 用這個檔案所在位置組路徑，不管從哪個工作目錄啟動伺服器都找得到
EMOTION_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "emotion2vec_plus_large")
# funasr 推論只需要這幾個檔案，README 和圖片不用下載
EMOTION_MODEL_FILES = ["config.yaml", "configuration.json", "model.pt", "tokens.txt"]

# 背景音樂分組：只看分數最高的情緒。
# 生氣/傷心/害怕 → negative (安撫型)，開心 → positive (輕快型)，
# 其他 (中性、驚訝、厭惡、other、<unk>) 不在表內 → None，不播背景音樂。
EMOTION_TO_GROUP = {
    "angry": "negative",
    "sad": "negative",
    "fearful": "negative",
    "happy": "positive",
}


def _ensure_model_files():
    if all(os.path.isfile(os.path.join(EMOTION_MODEL_DIR, f)) for f in EMOTION_MODEL_FILES):
        return
    print(f"[情緒模型] 找不到本機模型，從 HuggingFace 下載到 {EMOTION_MODEL_DIR} (約 1.9GB，只會下載這一次)")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=EMOTION_MODEL_ID, local_dir=EMOTION_MODEL_DIR, allow_patterns=EMOTION_MODEL_FILES)


def load_emotion_model(device: str):
    _ensure_model_files()
    # 載入時 funasr 會印出約 200 行逐層參數的 INFO log 與 decoder 的 "miss key" 警告
    # (emotion2vec 做分類用不到 decoder，缺了是正常的)，這裡暫時關掉避免洗版；
    # 載入失敗的例外照樣會往外丟，不會被吃掉。
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        logging.disable(logging.INFO)
        try:
            # funasr 匯入要好幾秒，放在函式裡，只有真的要載入模型時才付這個成本
            from funasr import AutoModel
            model = AutoModel(
                model=EMOTION_MODEL_DIR,   # 傳本機資料夾，funasr 就直接讀取不會再連網
                device=device,
                disable_update=True,   # 不要每次啟動都上網檢查 funasr 新版本
                disable_pbar=True,     # 不要每次推論都印進度條
            )
            # 暖機：第一次推論會多花約 0.2 秒初始化 CUDA kernel，在啟動時先付掉
            model.generate(np.zeros(16000, dtype=np.float32), granularity="utterance", extract_embedding=False)
        finally:
            logging.disable(logging.NOTSET)
    return model


def _label_to_en(label: str) -> str:
    # 模型的標籤長這樣：'生气/angry'、'难过/sad'、'<unk>'
    return label.split("/")[-1]


def detect_emotion(model, wav_16k: np.ndarray) -> dict:
    """
    wav_16k: 16kHz 單聲道的 float 波形 (就是 audio.load_wav(path, 16000) 的結果)

    回傳：
        {
          "top":    "sad",                          # 分數最高的單一情緒
          "scores": {"angry": 0.01, "sad": 0.82, ...},
          "group":  "negative" | "positive" | None,  # 背景音樂分組，None = 不播
        }
    """
    res = model.generate(
        wav_16k.astype(np.float32),
        granularity="utterance",   # 整段錄音給一個結果，不做逐幀
        extract_embedding=False,
    )[0]
    scores = {_label_to_en(l): float(s) for l, s in zip(res["labels"], res["scores"])}
    top = max(scores, key=scores.get)
    return {"top": top, "scores": scores, "group": EMOTION_TO_GROUP.get(top)}
