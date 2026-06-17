"""Centralized configuration for the Mandarin Pronunciation Coach.

All tunable constants live here. Module code reads from this file so that
sample rates, frame sizes, model identifiers, and weights can be adjusted
in one place.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------- paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
REF_TEXT_DIR: Path = DATA_DIR / "reference_texts"
STD_AUDIO_DIR: Path = DATA_DIR / "standard_audio"
CACHE_DIR: Path = DATA_DIR / "cache"
for _p in (DATA_DIR, REF_TEXT_DIR, STD_AUDIO_DIR, CACHE_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- audio / DSP
SAMPLE_RATE: int = 16_000           # all downstream code assumes 16 kHz mono
FRAME_LENGTH_MS: int = 25
HOP_LENGTH_MS: int = 10
FRAME_LENGTH: int = int(SAMPLE_RATE * FRAME_LENGTH_MS / 1000)   # 400
HOP_LENGTH: int = int(SAMPLE_RATE * HOP_LENGTH_MS / 1000)       # 160
PRE_EMPHASIS: float = 0.97
N_FFT: int = 512
N_MELS: int = 40
N_MFCC: int = 13

# F0 (librosa.pyin) — Mandarin speech is roughly 70~500 Hz across speakers
F0_MIN_HZ: float = 70.0
F0_MAX_HZ: float = 500.0

# VAD (webrtcvad) — aggressiveness 0..3, higher = more aggressive
VAD_AGGRESSIVENESS: int = 2
VAD_FRAME_MS: int = 30              # webrtcvad supports {10, 20, 30}

# ------------------------------------------------------------------- models
# Chinese wav2vec2 CTC checkpoint used for ASR + forced alignment + (opt.) GOP
WAV2VEC2_MODEL_ID: str = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
HF_CACHE_DIR: Path = CACHE_DIR / "hf"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
ALIYUN_ASR_MODEL: str = "qwen3-asr-flash"
ASR_ENGINE: str = "wav2vec2"  # Options: "wav2vec2", "aliyun-asr", or "auto".

# Device preference: "auto" -> cuda if available, else cpu
DEVICE: str = "auto"

# ------------------------------------------------------------------ scoring
# weights used by aggregator.aggregate()
DIM_WEIGHTS: dict = {
    "accuracy":     0.35,
    "tone":         0.25,
    "fluency":      0.15,
    "prosody":      0.15,
    "completeness": 0.10,
}

# Chao five-scale tone templates (target F0 contour shape, not absolute Hz)
# Values are 1..5 (low..high) sampled at uniform time steps.
TONE_TEMPLATES: dict = {
    1: [5, 5, 5, 5, 5],         # high level   55
    2: [3, 3, 4, 4, 5],         # mid rising   35
    3: [2, 1, 1, 3, 4],         # dipping      214
    4: [5, 4, 3, 2, 1],         # high falling 51
    5: [3, 2],                  # neutral (short, low-mid) — context dependent
}

# Number of samples each user syllable F0 contour is resampled to for matching.
TONE_RESAMPLE_LEN: int = 20

# Fluency targets (used to normalize raw values to 0..100 scores)
TARGET_SYLLABLES_PER_SEC: float = 4.0   # ~native casual speech rate
MAX_PAUSE_RATIO: float = 0.35           # > this fraction of silence -> 0 fluency

# ---------------------------------------------------------------- TTS engine
# Options: "edge-tts" (default), "aliyun-tts", "mimo-tts", or "f5-tts" (heavy, local).
TTS_ENGINE: str = "edge-tts"
EDGE_TTS_VOICE: str = "zh-CN-XiaoxiaoNeural"   # Mandarin female
ALIYUN_TTS_MODEL: str = "qwen3-tts-instruct-flash"
ALIYUN_TTS_VOICE: str = "Neil"
ALIYUN_TTS_LANGUAGE_TYPE: str = "Chinese"
ALIYUN_TTS_STYLE_PROMPT: str = (
    "请用普通话水平测试示范朗读的风格合成。要求字音准确，声调清楚，语速中等偏稳，"
    "语调平实端正，停顿自然，避免俏皮、撒娇、夸张表演和明显情绪化表达。"
)
MIMO_TTS_MODEL: str = "mimo-v2.5-tts"
MIMO_TTS_VOICE: str = "白桦"
MIMO_TTS_BASE_URL: str = "https://api.xiaomimimo.com/v1"
MIMO_TTS_STYLE_PROMPT: str = (
    "请用普通话水平测试示范朗读的风格合成。要求字音准确，声调清楚，语速中等偏稳，"
    "语调平实端正，停顿自然，避免俏皮、撒娇、夸张表演和明显情绪化表达。"
)
