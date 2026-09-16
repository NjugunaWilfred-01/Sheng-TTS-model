"""
Audio Preprocessing Pipeline: Normalization, Silero VAD Chunking, and Denoising.
Prepares raw Kenyan audio/podcasts into clean 3-15s speech segments for ASR/TTS training.
"""
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import soundfile as sf
import numpy as np
from pydub import AudioSegment

# Silero VAD - neural voice activity detection (replaces pydub silence detection)
import torch
from silero_vad import load_silero_vad, get_speech_timestamps, read_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AudioPreprocessor")


class ShengAudioPreprocessor:
    def __init__(
        self,
        target_sr: int = 16000,
        min_chunk_duration_sec: float = 2.0,
        max_chunk_duration_sec: float = 15.0,
        min_speech_duration_ms: int = 1500,
        max_speech_duration_s: float = 15.0,
        min_silence_duration_ms: int = 400
    ):
        self.target_sr = target_sr
        self.min_chunk_sec = min_chunk_duration_sec
        self.max_chunk_sec = max_chunk_duration_sec
        self.min_speech_duration_ms = min_speech_duration_ms
        self.max_speech_duration_s = max_speech_duration_s
        self.min_silence_duration_ms = min_silence_duration_ms
        # Load Silero VAD model once
        self.vad_model = load_silero_vad()

    def normalize_audio(self, audio_path: str, output_path: str = None) -> str:
        """Converts audio to 16kHz Mono PCM WAV and normalizes volume."""
        if not output_path:
            output_path = str(Path(audio_path).with_suffix(".norm.wav"))

        audio = AudioSegment.from_file(audio_path)
        # Convert to Mono, 16000Hz, 16-bit PCM
        audio = audio.set_channels(1).set_frame_rate(self.target_sr).set_sample_width(2)
        # Normalize volume to -20 dBFS
        change_in_db = -20.0 - audio.dBFS
        audio = audio.apply_gain(change_in_db)
        audio.export(output_path, format="wav")
        logger.info(f"Normalized audio saved to: {output_path}")
        return output_path

    def chunk_audio_by_vad(self, audio_path: str, output_dir: str) -> List[Dict[str, Any]]:
        """
        Splits long audio into conversational turns/chunks using Silero VAD.
        Silero is a neural VAD that detects speech, not loudness.
        Ensures all chunks are between min_chunk_sec and max_chunk_sec.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Read audio with Silero's helper (resamples to target_sr automatically)
        wav = read_audio(audio_path, sampling_rate=self.target_sr)
        
        # Get speech timestamps from Silero VAD
        timestamps = get_speech_timestamps(
            wav,
            self.vad_model,
            sampling_rate=self.target_sr,
            min_speech_duration_ms=self.min_speech_duration_ms,
            max_speech_duration_s=self.max_speech_duration_s,
            min_silence_duration_ms=self.min_silence_duration_ms,
            return_seconds=True,
        )

        processed = []
        for idx, ts in enumerate(timestamps):
            start, end = int(ts["start"] * self.target_sr), int(ts["end"] * self.target_sr)
            segment = wav[start:end].numpy()
            dur = (end - start) / self.target_sr
            
            # Skip chunks outside valid duration range
            if dur < self.min_chunk_sec or dur > self.max_chunk_sec:
                continue
            
            path = out_dir / f"chunk_{idx:05d}.wav"
            sf.write(str(path), segment, self.target_sr)
            processed.append({
                "chunk_id": idx,
                "file_path": str(path),
                "duration_sec": round(dur, 2)
            })

        logger.info(f"Generated {len(processed)} speech chunks in {output_dir}")
        return processed
