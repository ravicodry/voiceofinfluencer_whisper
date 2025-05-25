# src/whisper_utils.py

import os
import subprocess
from faster_whisper import WhisperModel

# (keep download_audio, convert_to_wav, cleanup_temp_files exactly the same)

# Cache model loads so your app’s @st.cache_resource still works
_model_cache = {}

def load_whisper_model(model_size: str):
    if model_size not in _model_cache:
        # device="cpu" is default
        _model_cache[model_size] = WhisperModel(model_size, device="cpu")
    return _model_cache[model_size]

def transcribe_audio(model, wav_path: str) -> str:
    # faster-whisper returns (segments, info)
    segments, _ = model.transcribe(wav_path, beam_size=5)
    # Join all segment texts into one transcript string
    return "".join(seg.text for seg in segments)
