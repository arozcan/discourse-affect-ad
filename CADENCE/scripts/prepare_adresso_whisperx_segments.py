#!/usr/bin/env python
"""
ADReSSo21 için WhisperX tabanlı hazırlık script'i (CSV ve diarization YOK).

Yeni alternatif mantık:

- CSV segmentation tamamen YOK SAYILIR.
- Tüm ses dosyası baştan sona WhisperX ASR ile çözülür, word-level timestamplar alınır.
- WhisperX segmentleri kullanılmaz, bunun yerine kelime zamanlamalarına göre ikincil segmentasyon yapılır.
- Segmentler doğal duraklamalara (kelimeler arası 0.6 sn üzeri boşluk) ve maksimum/minimum segment uzunluklarına göre oluşturulur.
- Tüm segmentler konuşmacı olarak PAR kabul edilir.
- Diarization kullanılmaz.

Çıktı JSON formatı:
{
  "id": "adrso018",
  "split": "train" | "test",
  "label": "ad" | "cn" | null,
  "audio": "...wav",
  "segments": [
    {
      "speaker": "PAR",
      "start_ms": ...,
      "end_ms": ...,
      "text": "..."
    },
    ...
  ]
}
"""

import os
import glob
import json
import re
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

import torch
import whisperx
import argparse

# === Kök klasörler ===
ADRESSO_ROOT = "download/ADReSSo21/diagnosis"
OUTPUT_ROOT = "dataset/ADReSSo21_WHISPERX_SEG"

# ------------------------------------------------------------------
# 1) Cihaz + WhisperX ASR modeli
# ------------------------------------------------------------------
if torch.cuda.is_available():
    device = "cuda"
    compute_type = "float16"  # GPU varsa daha hızlı
else:
    device = "cpu"
    compute_type = "int8"     # CPU'da bellek için daha hafif (istersen "float32" yapabilirsin)

print(f"▶ Torch device: {device}")
print(f"▶ WhisperX ASR modeli yükleniyor (large-v2, device={device}, compute_type={compute_type})")

asr_model = whisperx.load_model(
    "large-v2",
    device=device,
    compute_type=compute_type,
)

# Yeni segmentasyon parametreleri
PAUSE_SPLIT_SEC = 0.9  # Kelimeler arası boşluk 0.9 sn veya daha fazla ise yeni segment başlat
MAX_SEG_SEC = 6.0
MIN_SEG_SEC = 0.8


# ------------------------------------------------------------------
# 2) Yardımcı fonksiyonlar
# ------------------------------------------------------------------

def transcribe_interval(
    audio: np.ndarray,
    sr: int,
    start_ms: float,
    end_ms: float,
    language: str = "en",
    pad_before_ms: float = 50.0,
    pad_after_ms: float = 50.0,
    min_duration_ms: float = 200.0,
) -> List[Dict[str, Any]]:
    """ASR on [start_ms, end_ms] using WhisperX segment timestamps.

    Returns a list of segments with their own start/end times (ms) and text,
    where times are derived from WhisperX's segment timestamps and mapped back
    to the global timeline. Optional padding is applied only for ASR but
    segment times are clamped back to the original [start_ms, end_ms].
    """
    n_samples = len(audio)
    total_ms = n_samples * 1000.0 / sr

    # Padded interval for ASR
    s_ms_pad = max(0.0, start_ms - pad_before_ms)
    e_ms_pad = min(total_ms, end_ms + pad_after_ms)
    if e_ms_pad <= s_ms_pad:
        return []

    s_idx = int(s_ms_pad / 1000.0 * sr)
    e_idx = int(e_ms_pad / 1000.0 * sr)
    s_idx = max(0, min(n_samples, s_idx))
    e_idx = max(0, min(n_samples, e_idx))
    if e_idx <= s_idx:
        return []

    seg_audio = audio[s_idx:e_idx]

    # Çok kısa ise (örn. < min_duration_ms) atla
    if e_ms_pad - s_ms_pad < min_duration_ms:
        return []

    # 16 kHz'e resample
    target_sr = 16000
    if sr != target_sr:
        seg_audio = resample_poly(seg_audio, target_sr, sr)
    seg_audio = seg_audio.astype("float32")

    try:
        result = asr_model.transcribe(
            seg_audio,
            batch_size=8,
            language=language,
        )
    except Exception as e:
        print(f"    ❌ ASR hatası [{start_ms:.0f}-{end_ms:.0f} ms]: {e}")
        return []

    out_segments: List[Dict[str, Any]] = []
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        local_start_ms = float(seg.get("start", 0.0)) * 1000.0
        local_end_ms = float(seg.get("end", 0.0)) * 1000.0

        # Haritalama: padded interval başlangıcına göre, sonra orijinale clamp
        g_start = s_ms_pad + local_start_ms
        g_end = s_ms_pad + local_end_ms

        # Orijinal CSV/gap aralığına sıkıştır
        g_start = max(start_ms, min(g_start, end_ms))
        g_end = max(start_ms, min(g_end, end_ms))
        if g_end <= g_start:
            continue

        out_segments.append(
            {
                "start_ms": g_start,
                "end_ms": g_end,
                "text": text,
            }
        )

    return out_segments


def process_file(
    audio_path: str,
    seg_path: str,
    label: str,
    split: str,
    pad_before_ms: float = 50.0,
    pad_after_ms: float = 50.0,
) -> Dict[str, Any] | None:
    """
    Tüm ses dosyasını WhisperX ile çöz, segmentleri doğal duraklamalara göre oluştur.
    Tüm segmentler PAR olarak kabul edilir.
    """
    try:
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
    except Exception as e:
        print(f"❌ Audio okuma hatası: {audio_path} -> {e}")
        return None

    print(f"▶ Full-audio ASR (no CSV): {os.path.basename(audio_path)}")

    segments = transcribe_full_audio(
        audio=audio,
        sr=sr,
        language="en",
    )

    if not segments:
        print(f"⚠️ Hiç segment üretilemedi: {audio_path}")
        return None

    file_id = os.path.splitext(os.path.basename(audio_path))[0]
    return {
        "id": file_id,
        "split": split,
        "label": label,
        "audio": audio_path,
        "segments": segments,
    }


# Yeni yardımcı fonksiyon: Full-audio ASR using WhisperX word-level timestamps and secondary segmentation
def transcribe_full_audio(
    audio: np.ndarray,
    sr: int,
    language: str = "en",
) -> List[Dict[str, Any]]:
    """
    Full-audio ASR using WhisperX with word-level timestamps.
    Secondary pause-based segmentation is applied:
    - New segment if gap between words >= PAUSE_SPLIT_SEC or current segment duration >= MAX_SEG_SEC
    - Segments shorter than MIN_SEG_SEC are discarded or merged.
    All speakers are labeled as PAR.
    """
    # Resample to 16 kHz if needed
    target_sr = 16000
    if sr != target_sr:
        audio = resample_poly(audio, target_sr, sr)
        sr = target_sr

    audio = audio.astype("float32")

    try:
        result = asr_model.transcribe(
            audio,
            batch_size=8,
            language=language,
        )
    except Exception as e:
        print(f"❌ ASR hatası (full audio): {e}")
        return []

    # Alignment modeli yükle
    align_model, metadata = whisperx.load_align_model(
        language_code=language,
        device=device
    )
    aligned_result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False
    )

    # Flatten all words from all segments preserving global start/end
    words = []
    for seg in aligned_result.get("segments", []):
        for w in seg.get("words", []):
            # Each word dict should have 'word', 'start', 'end'
            if 'word' in w and 'start' in w and 'end' in w:
                words.append({
                    "word": w["word"],
                    "start": w["start"],
                    "end": w["end"],
                })

    if not words:
        return []

    segments = []
    current_segment_words = []
    current_segment_start = words[0]["start"]
    current_segment_end = words[0]["end"]

    for i in range(len(words)):
        w = words[i]
        if not current_segment_words:
            current_segment_words.append(w)
            current_segment_start = w["start"]
            current_segment_end = w["end"]
            continue

        prev_word = current_segment_words[-1]
        gap = w["start"] - prev_word["end"]

        segment_duration = current_segment_end - current_segment_start

        # --- Soft split logic ---
        soft_split = False
        if gap >= PAUSE_SPLIT_SEC:
            soft_split = True
        elif segment_duration >= MAX_SEG_SEC and gap >= 0.3:
            soft_split = True

        # Check if need to start a new segment
        if soft_split:
            # Close current segment if long enough
            segment_duration = current_segment_end - current_segment_start
            if segment_duration >= MIN_SEG_SEC:
                segment_text = " ".join([ww["word"] for ww in current_segment_words]).strip()
                segments.append({
                    "speaker": "PAR",
                    "start_ms": current_segment_start * 1000.0,
                    "end_ms": current_segment_end * 1000.0,
                    "text": segment_text,
                })
                # Start new segment
                current_segment_words = [w]
                current_segment_start = w["start"]
                current_segment_end = w["end"]
            else:
                # Segment too short: merge with next segment by continuing
                current_segment_words.append(w)
                current_segment_end = w["end"]
        else:
            current_segment_words.append(w)
            current_segment_end = w["end"]

    # Add last segment if long enough
    if current_segment_words:
        segment_duration = current_segment_end - current_segment_start
        if segment_duration >= MIN_SEG_SEC:
            segment_text = " ".join([ww["word"] for ww in current_segment_words]).strip()
            segments.append({
                "speaker": "PAR",
                "start_ms": current_segment_start * 1000.0,
                "end_ms": current_segment_end * 1000.0,
                "text": segment_text,
            })
        else:
            # If last segment too short and there is a previous segment, merge it
            if segments:
                segments[-1]["end_ms"] = current_segment_end * 1000.0
                segments[-1]["text"] += " " + " ".join([ww["word"] for ww in current_segment_words]).strip()
            else:
                # No previous segment, add anyway
                segment_text = " ".join([ww["word"] for ww in current_segment_words]).strip()
                segments.append({
                    "speaker": "PAR",
                    "start_ms": current_segment_start * 1000.0,
                    "end_ms": current_segment_end * 1000.0,
                    "text": segment_text,
                })

    return segments


# ------------------------------------------------------------------
# 3) train ve test split'leri
# ------------------------------------------------------------------
def build_split_train():
    """
    train:
      audio:        ADReSSo21/diagnosis/train/audio/{ad,cn}/*.wav
    -->
      OUTPUT_ROOT/train/*.json
    """
    split = "train"
    audio_root = os.path.join(ADRESSO_ROOT, "train", "audio")
    out_dir = os.path.join(OUTPUT_ROOT, "train")
    os.makedirs(out_dir, exist_ok=True)

    index = []

    for label in ["ad", "cn"]:
        audio_dir = os.path.join(audio_root, label)
        if not os.path.isdir(audio_dir):
            print(f"⚠️ Audio klasörü yok, atlanıyor: {audio_dir}")
            continue

        audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
        print(f"📂 train / {label}: {len(audio_files)} audio dosyası")

        for audio_path in audio_files:
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            print(f"▶ İşleniyor (train {label}): {audio_path}")
            data = process_file(
                audio_path=audio_path,
                seg_path=None,
                label=label,
                split=split,
            )

            if data is None:
                continue

            out_path = os.path.join(out_dir, base_name + ".json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            index.append(
                {
                    "id": data["id"],
                    "split": split,
                    "label": data["label"],
                    "json_path": out_path,
                }
            )

    index_path = os.path.join(out_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\n✅ TRAIN bitti: {len(index)} dosya işlendi → {out_dir}")
    print(f"📄 Index dosyası: {index_path}")


def build_split_test():
    """
    test-dist:
      audio:        ADReSSo21/diagnosis/test-dist/audio/*.wav
    -->
      OUTPUT_ROOT/test/*.json
    """
    split = "test"
    audio_dir = os.path.join(ADRESSO_ROOT, "test-dist", "audio")
    out_dir = os.path.join(OUTPUT_ROOT, "test")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(audio_dir):
        print(f"⚠️ Test audio klasörü yok: {audio_dir}")
        return

    audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    print(f"📂 test-dist: {len(audio_files)} audio dosyası")

    index = []

    for audio_path in audio_files:
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        print(f"▶ İşleniyor (test): {audio_path}")
        data = process_file(
            audio_path=audio_path,
            seg_path=None,
            label=None,  # test label yok
            split=split,
        )

        if data is None:
            continue

        out_path = os.path.join(out_dir, base_name + ".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        index.append(
            {
                "id": data["id"],
                "split": split,
                "label": None,
                "json_path": out_path,
            }
        )

    index_path = os.path.join(out_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\n✅ TEST bitti: {len(index)} dosya işlendi → {out_dir}")
    print(f"📄 Index dosyası: {index_path}")


# ------------------------------------------------------------------
# 4) main
# ------------------------------------------------------------------
def main():
    global ADRESSO_ROOT, OUTPUT_ROOT

    parser = argparse.ArgumentParser(
        description="Prepare ADReSSo21 WhisperX-based segmented dataset"
    )
    parser.add_argument(
        "--adresso_root",
        type=str,
        default="download/ADReSSo21/diagnosis",
        help="ADReSSo21 diagnosis root directory"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="dataset/ADReSSo21_WHISPERX_SEG",
        help="Output directory for WhisperX-segmented dataset"
    )

    args = parser.parse_args()

    ADRESSO_ROOT = args.adresso_root
    OUTPUT_ROOT = args.output_root

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    build_split_train()
    #build_split_test()


if __name__ == "__main__":
    main()