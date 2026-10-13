"""
背景音樂：依情緒分組挑歌、計算音量、產生 ffmpeg 混音濾鏡。

用法：
    bgm = pick_bgm(emotion, wav_16k)     # emotion 是 detect_emotion() 的結果；不需要配樂時回傳 None
    filter = build_mix_filter(bgm["gain_db"], duration)   # 給 ffmpeg -filter_complex 用
"""
import os
import random
import subprocess
from functools import lru_cache

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MUSIC_DIRS = {
    "negative": os.path.join(BASE_DIR, "music", "negative(sad_angry)"),
    "positive": os.path.join(BASE_DIR, "music", "positive(happy)"),
}
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac")

# 音量是「相對於人聲」設定的，而不是固定倍率：
# 兩首歌原始音量就差了約 7dB (烏克麗麗那首比較大聲)，固定倍率會一首太吵一首聽不到，
# 所以每次都量人聲與音樂的實際音量，把音樂調到「比人聲小 BGM_BELOW_SPEECH_DB」。
#   16dB → 音樂振幅約為人聲的 16%，墊在底下、不搶話
#   生氣時再加 ANGRY_BOOST_DB → 比人聲小 13dB，振幅約 22%
BGM_BELOW_SPEECH_DB = 16.0
ANGRY_BOOST_DB = 3.0

# 只用「有在說話」的片段算人聲音量，停頓的靜音不算進去，
# 否則停頓多的錄音會被低估音量，音樂就會被調得相對太大聲
_FRAME_SEC = 0.05
_SILENCE_DB = -45.0


def _active_level_db(x: np.ndarray, sr: int = 16000) -> float:
    n = int(sr * _FRAME_SEC)
    frames = x[: len(x) // n * n].reshape(-1, n) if len(x) >= n else x.reshape(1, -1)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    active = rms[rms > 10 ** (_SILENCE_DB / 20)]
    if active.size == 0:
        active = rms
    return float(20 * np.log10(np.sqrt(np.mean(active ** 2)) + 1e-12))


@lru_cache(maxsize=32)
def _music_samples(path: str, mtime: float) -> np.ndarray:
    # 解碼成 16kHz 單聲道波形 (一首 2 分鐘的歌約 8MB) 快取起來，每首歌只需解碼一次；
    # mtime 放進快取 key：同名檔案被換掉時會重新解碼
    raw = subprocess.run(
        ['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', '16000', '-f', 'f32le', '-'],
        stdout=subprocess.PIPE, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def _played_music_level_db(path: str, n_samples: int) -> float:
    # 只量「這次實際會播到的那一段」：錄音通常只有幾秒，只會播到歌的開頭，
    # 而前奏常比整首平均安靜 (烏克麗麗那首的開頭就安靜了約 4dB)，用整首平均會把音樂壓得太小聲。
    # np.resize 會循環填滿，剛好對應 ffmpeg -stream_loop 的循環播放。
    music = _music_samples(path, os.path.getmtime(path))
    return _active_level_db(np.resize(music, n_samples))


def pick_bgm(emotion: dict, wav_16k: np.ndarray):
    """
    回傳 {"path": 音樂檔, "gain_db": 要套用的增益, "group": "negative"/"positive"}；
    情緒不需要配樂、或資料夾裡沒有音樂時回傳 None。
    """
    group = emotion.get("group")
    if group is None:
        return None

    folder = MUSIC_DIRS[group]
    files = sorted(
        os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(AUDIO_EXTS)
    ) if os.path.isdir(folder) else []
    if not files:
        print(f"[配樂] {folder} 裡沒有音樂檔，這次不加背景音樂")
        return None

    path = random.choice(files)   # 目前每個資料夾只有一首；之後放多首會隨機挑
    target_db = _active_level_db(wav_16k) - BGM_BELOW_SPEECH_DB
    if emotion.get("top") == "angry":
        target_db += ANGRY_BOOST_DB
    gain_db = target_db - _played_music_level_db(path, len(wav_16k))
    gain_db = float(np.clip(gain_db, -40.0, 10.0))   # 防呆：極端錄音時不要放大或壓低到離譜
    return {"path": path, "gain_db": gain_db, "group": group}


def build_mix_filter(gain_db: float, duration: float) -> str:
    """
    ffmpeg -filter_complex 的音訊部分。輸入約定：[1:a] = 人聲、[2:a] = 背景音樂 (已用 -stream_loop -1 無限循環)。
    輸出標籤 [aout]。

    - 人聲與音樂都轉成 44.1kHz 立體聲再混，音樂才不會被降到人聲的 16kHz 而變悶
    - amix duration=first：以人聲長度為準，音樂循環播放、在說話結束時一起結束
    - 開頭淡入、結尾淡出，避免音樂突然出現或被硬切斷 (短錄音時按比例縮短)
    - normalize=0：不要讓 amix 自動把兩軌各減半，人聲維持原本音量
    - alimiter：人聲峰值加上音樂時可能超過 0dB，壓一下避免爆音 (level=false 不做自動增益)
    """
    fade_in = min(0.5, duration / 4)
    fade_out = min(1.0, duration / 3)
    fade_out_start = max(0.0, duration - fade_out)
    return (
        "[1:a]aformat=sample_rates=44100:channel_layouts=stereo[voice];"
        "[2:a]aformat=sample_rates=44100:channel_layouts=stereo,"
        f"volume={gain_db:.2f}dB,"
        f"afade=t=in:d={fade_in:.2f},"
        f"afade=t=out:st={fade_out_start:.2f}:d={fade_out:.2f}[bgm];"
        "[voice][bgm]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
        "alimiter=limit=0.95:level=false[aout]"
    )
