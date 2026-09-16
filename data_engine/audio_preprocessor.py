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
from pydub import AudioSegment, silence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AudioPreprocessor")


class ShengAudioPreprocessor:
    def __init__(
        self,
        target_sr: int = 16000,
        min_chunk_duration_sec: float = 2.0,
        max_chunk_duration_sec: float = 15.0,
        silence_thresh_db: int = -36,
        min_silence_len_ms: int = 400
    ):
        self.target_sr = target_sr
        self.min_chunk_sec = min_chunk_duration_sec
        self.max_chunk_sec = max_chunk_duration_sec
        self.silence_thresh_db = silence_thresh_db
        self.min_silence_len_ms = min_silence_len_ms

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
        Split long audio into conversational turns.

        Prefers Silero VAD, a neural model that detects SPEECH. Falls back to pydub's
        energy-gate splitter only if silero-vad is not installed.

        The distinction matters more than it looks. pydub cuts on raw loudness, so a
        speaker drawing breath mid-sentence reads as silence and the word gets cut in
        half, while two people talking over a quiet gap get merged into one chunk.
        Every chunk downstream -- pseudo-label, hand correction, fine-tune target --
        inherits that error. Silero cuts on whether speech is present, which is the
        actual question being asked.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            return self._chunk_with_silero(audio_path, out_dir)
        except ImportError:
            logger.warning(
                "silero-vad not installed - falling back to pydub energy gate, which "
                "cuts mid-word and merges speakers. Install with: pip install silero-vad"
            )
        except Exception as e:
            logger.warning(f"Silero VAD failed ({e}); falling back to pydub energy gate.")

        return self._chunk_with_pydub(audio_path, out_dir)

    def _chunk_with_silero(self, audio_path: str, out_dir: Path) -> List[Dict[str, Any]]:
        """Neural VAD chunking. Raises ImportError if silero-vad is unavailable."""
        from silero_vad import load_silero_vad, get_speech_timestamps, read_audio

        wav = read_audio(str(audio_path), sampling_rate=self.target_sr)
        model = load_silero_vad()
        timestamps = get_speech_timestamps(
            wav, model,
            sampling_rate=self.target_sr,
            min_speech_duration_ms=1500,
            max_speech_duration_s=self.max_chunk_sec,
            min_silence_duration_ms=400,
            return_seconds=True,
        )

        processed_chunks = []
        for idx, ts in enumerate(timestamps):
            start = int(ts["start"] * self.target_sr)
            end = int(ts["end"] * self.target_sr)
            duration_sec = (end - start) / self.target_sr
            if not (self.min_chunk_sec <= duration_sec <= self.max_chunk_sec):
                continue
            chunk_path = out_dir / f"chunk_{len(processed_chunks):05d}.wav"
            sf.write(str(chunk_path), wav[start:end].numpy(), self.target_sr)
            processed_chunks.append({
                "chunk_id": len(processed_chunks),
                "file_path": str(chunk_path),
                "duration_sec": round(duration_sec, 2),
            })

        logger.info(f"Silero VAD: {len(processed_chunks)} speech chunks in {out_dir}")
        return processed_chunks

    def _chunk_with_pydub(self, audio_path: str, out_dir: Path) -> List[Dict[str, Any]]:
        """Energy-gate fallback. Less accurate -- see chunk_audio_by_vad docstring."""
        audio = AudioSegment.from_file(audio_path)
        chunks = silence.split_on_silence(
            audio,
            min_silence_len=self.min_silence_len_ms,
            silence_thresh=self.silence_thresh_db,
            keep_silence=200
        )

        processed_chunks = []
        accumulated_segment = AudioSegment.empty()
        chunk_idx = 0

        for segment in chunks:
            duration_sec = len(segment) / 1000.0

            # If segment is too short, accumulate it
            if len(accumulated_segment) / 1000.0 + duration_sec < self.max_chunk_sec:
                accumulated_segment += segment
            else:
                # Save accumulated segment if within valid range
                if len(accumulated_segment) / 1000.0 >= self.min_chunk_sec:
                    chunk_filename = f"chunk_{chunk_idx:05d}.wav"
                    chunk_path = out_dir / chunk_filename
                    accumulated_segment.export(str(chunk_path), format="wav")
                    processed_chunks.append({
                        "chunk_id": chunk_idx,
                        "file_path": str(chunk_path),
                        "duration_sec": round(len(accumulated_segment) / 1000.0, 2)
                    })
                    chunk_idx += 1
                accumulated_segment = segment

        # Handle remaining segment
        if len(accumulated_segment) / 1000.0 >= self.min_chunk_sec:
            chunk_filename = f"chunk_{chunk_idx:05d}.wav"
            chunk_path = out_dir / chunk_filename
            accumulated_segment.export(str(chunk_path), format="wav")
            processed_chunks.append({
                "chunk_id": chunk_idx,
                "file_path": str(chunk_path),
                "duration_sec": round(len(accumulated_segment) / 1000.0, 2)
            })

        logger.info(f"pydub energy gate: {len(processed_chunks)} chunks in {out_dir}")
        return processed_chunks
