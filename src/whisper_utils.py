# src/whisper_utils.py

import os
import subprocess
from faster_whisper import WhisperModel

# Download audio (unchanged)
def download_audio(url: str, out_template: str = "temp.%(ext)s") -> str:
    from yt_dlp import YoutubeDL
    opts = {"format":"bestaudio","outtmpl":out_template,"quiet":True,"no_warnings":True}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return f"temp.{info['ext']}"

# Convert to WAV (unchanged)
def convert_to_wav(input_file: str, output_file: str = "audio.wav") -> str:
    cmd = ["ffmpeg","-i",input_file,"-ar","16000","-ac","1",output_file,"-y"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_file

# Clean up temps (unchanged)
def cleanup_temp_files():
    for f in ("temp.m4a","temp.webm","audio.wav"):
        if os.path.exists(f):
            os.remove(f)

# Cache models so load only once
_model_cache: dict[str, WhisperModel] = {}

def load_whisper_model(model_size: str) -> WhisperModel:
    """Return a faster-whisper model, loading and caching it per size."""
    if model_size not in _model_cache:
        # device="cpu" is default
        _model_cache[model_size] = WhisperModel(model_size, device="cpu")
    return _model_cache[model_size]

def transcribe_audio(model: WhisperModel, wav_path: str) -> str:
    """Run transcription and concatenate all segments."""
    segments, _ = model.transcribe(wav_path, beam_size=5)
    return "".join(seg.text for seg in segments)
