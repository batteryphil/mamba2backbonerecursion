"""Data pipeline — download, process, and load training data.

Handles three datasets:
    1. Jenny (Dioco) — 30h British/Irish female (base voice)
    2. EMNS (SLR136) — 2.3h British female with emotion labels
    3. VCTK — select British female speakers (accent reinforcement)

Extracts mel spectrograms, pitch (F0), energy, and phoneme durations.
"""

import csv
import json
import os
import subprocess
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import librosa
except ImportError:
    raise ImportError("librosa required: pip install librosa")

from text_frontend import TextFrontend, SYMBOL_TO_ID


# ─── Audio Feature Extraction ───────────────────────────────────────────────


class AudioProcessor:
    """Extract mel spectrogram, pitch, and energy from audio.

    Args:
        sample_rate: Target sample rate.
        n_fft: FFT window size.
        hop_length: Hop size between frames.
        win_length: Window length.
        n_mels: Number of mel frequency bins.
        fmin: Minimum frequency.
        fmax: Maximum frequency.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: float = 8000.0,
    ) -> None:
        """Initialize AudioProcessor."""
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

    def load_audio(self, path: str) -> np.ndarray:
        """Load audio file and resample to target rate.

        Args:
            path: Path to audio file.

        Returns:
            Audio waveform as numpy array.
        """
        audio, sr = librosa.load(path, sr=self.sample_rate)
        return audio

    def extract_mel(self, audio: np.ndarray) -> np.ndarray:
        """Extract normalized mel spectrogram.

        Args:
            audio: Audio waveform.

        Returns:
            Mel spectrogram [T, n_mels].
        """
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        mel_db = librosa.power_to_db(mel, ref=20.0, top_db=100.0)
        # Normalize to [0, 1]
        mel_norm = (mel_db + 100.0) / 100.0
        return mel_norm.T  # [T, n_mels]

    def extract_pitch(self, audio: np.ndarray) -> np.ndarray:
        """Extract F0 (fundamental frequency) contour using pyin.

        Args:
            audio: Audio waveform.

        Returns:
            F0 contour [T_frames].
        """
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )
        # Replace NaN (unvoiced) with 0
        f0 = np.nan_to_num(f0, nan=0.0)
        return f0

    def extract_energy(self, audio: np.ndarray) -> np.ndarray:
        """Extract per-frame energy (RMS).

        Args:
            audio: Audio waveform.

        Returns:
            Energy contour [T_frames].
        """
        energy = librosa.feature.rms(
            y=audio,
            frame_length=self.win_length,
            hop_length=self.hop_length,
        )[0]
        return energy


# ─── Duration Extraction ────────────────────────────────────────────────────


def estimate_durations(
    n_phonemes: int,
    n_mel_frames: int,
) -> np.ndarray:
    """Estimate phoneme durations using uniform distribution.

    This is a simple heuristic that distributes mel frames
    evenly across phonemes. For production quality, use
    Montreal Forced Aligner instead.

    Args:
        n_phonemes: Number of phonemes.
        n_mel_frames: Number of mel frames.

    Returns:
        Duration array [n_phonemes].
    """
    base_dur = n_mel_frames // n_phonemes
    remainder = n_mel_frames % n_phonemes
    durations = np.full(n_phonemes, base_dur, dtype=np.int64)
    # Distribute remainder across first N phonemes
    durations[:remainder] += 1
    return durations


# ─── Dataset Download / Preparation ─────────────────────────────────────────


def download_jenny(data_dir: str = "data") -> str:
    """Download and extract the Jenny (Dioco) dataset.

    Args:
        data_dir: Root data directory.

    Returns:
        Path to extracted Jenny dataset directory.
    """
    jenny_dir = os.path.join(data_dir, "jenny")
    if os.path.exists(jenny_dir) and os.listdir(jenny_dir):
        print(f"[INFO] Jenny dataset already exists at {jenny_dir}")
        return jenny_dir

    os.makedirs(jenny_dir, exist_ok=True)
    url = "https://www.languagereactor.com/dataset.tar.zst"
    archive_path = os.path.join(data_dir, "jenny_dataset.tar.zst")

    print("[INFO] Downloading Jenny (Dioco) dataset (~30h)...")
    print(f"  URL: {url}")
    subprocess.run(
        ["wget", "-c", "-O", archive_path, url],
        check=True,
    )

    print("[INFO] Extracting Jenny dataset...")
    subprocess.run(
        ["tar", "--zstd", "-xf", archive_path, "-C", jenny_dir],
        check=True,
    )

    print(f"[INFO] Jenny dataset ready at {jenny_dir}")
    return jenny_dir


def download_emns(data_dir: str = "data") -> str:
    """Download and extract the EMNS corpus (SLR136).

    Args:
        data_dir: Root data directory.

    Returns:
        Path to extracted EMNS dataset directory.
    """
    emns_dir = os.path.join(data_dir, "emns")
    if os.path.exists(emns_dir) and os.listdir(emns_dir):
        print(f"[INFO] EMNS dataset already exists at {emns_dir}")
        return emns_dir

    os.makedirs(emns_dir, exist_ok=True)
    base_url = "https://www.openslr.org/resources/136"

    files = [
        "cleaned_webm.tar.xz",
        "cleaned_alignment.tar.xz",
        "metadata.csv",
    ]

    for fname in files:
        url = f"{base_url}/{fname}"
        out_path = os.path.join(emns_dir, fname)
        print(f"[INFO] Downloading EMNS: {fname}...")
        subprocess.run(
            ["wget", "-c", "-O", out_path, url],
            check=True,
        )
        if fname.endswith(".tar.xz"):
            print(f"[INFO] Extracting {fname}...")
            subprocess.run(
                ["tar", "-xf", out_path, "-C", emns_dir],
                check=True,
            )

    print(f"[INFO] EMNS dataset ready at {emns_dir}")
    return emns_dir


def download_vctk(data_dir: str = "data") -> str:
    """Download VCTK dataset (subset of British female speakers).

    Args:
        data_dir: Root data directory.

    Returns:
        Path to VCTK dataset directory.
    """
    vctk_dir = os.path.join(data_dir, "vctk")
    if os.path.exists(vctk_dir) and os.listdir(vctk_dir):
        print(f"[INFO] VCTK dataset already exists at {vctk_dir}")
        return vctk_dir

    os.makedirs(vctk_dir, exist_ok=True)
    url = (
        "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/"
        "VCTK-Corpus-0.92.zip"
    )

    archive_path = os.path.join(data_dir, "VCTK-Corpus-0.92.zip")
    print("[INFO] Downloading VCTK Corpus...")
    subprocess.run(
        ["wget", "-c", "-O", archive_path, url],
        check=True,
    )

    print("[INFO] Extracting VCTK...")
    subprocess.run(
        ["unzip", "-o", archive_path, "-d", vctk_dir],
        check=True,
    )

    print(f"[INFO] VCTK dataset ready at {vctk_dir}")
    return vctk_dir


# British female speakers from VCTK (Southern England accents)
VCTK_BRITISH_FEMALE_SPEAKERS = [
    "p225",  # Southern England, female
    "p228",  # Southern England, female
    "p229",  # Southern England, female
    "p236",  # Manchester, female
    "p239",  # Southeast England, female
]


# ─── PyTorch Dataset ─────────────────────────────────────────────────────────


class TTSDataset(Dataset):
    """TTS dataset yielding (phonemes, mel, pitch, energy, duration, emotion).

    Args:
        manifest_path: Path to JSON manifest file.
        audio_processor: AudioProcessor instance.
        text_frontend: TextFrontend instance.
        max_mel_len: Maximum mel spectrogram length.
    """

    def __init__(
        self,
        manifest_path: str,
        audio_processor: AudioProcessor,
        text_frontend: TextFrontend,
        max_mel_len: int = 1000,
    ) -> None:
        """Initialize TTSDataset."""
        self.audio_processor = audio_processor
        self.text_frontend = text_frontend
        self.max_mel_len = max_mel_len

        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        print(f"[INFO] Loaded {len(self.manifest)} samples from {manifest_path}")

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample.

        Args:
            idx: Sample index.

        Returns:
            Dict with phonemes, mel, pitch, energy, duration, emotion_id.
        """
        item = self.manifest[idx]

        # Load and process audio
        audio = self.audio_processor.load_audio(item["audio_path"])
        mel = self.audio_processor.extract_mel(audio)
        pitch = self.audio_processor.extract_pitch(audio)
        energy = self.audio_processor.extract_energy(audio)

        # Text to phonemes
        phoneme_ids = self.text_frontend.text_to_ids(item["text"])

        # Align lengths
        n_mel = mel.shape[0]
        n_pitch = len(pitch)
        n_energy = len(energy)
        min_len = min(n_mel, n_pitch, n_energy)
        mel = mel[:min_len]
        pitch = pitch[:min_len]
        energy = energy[:min_len]

        # Truncate if too long
        if min_len > self.max_mel_len:
            mel = mel[:self.max_mel_len]
            pitch = pitch[:self.max_mel_len]
            energy = energy[:self.max_mel_len]
            min_len = self.max_mel_len

        # Estimate durations
        durations = estimate_durations(len(phoneme_ids), min_len)

        # Emotion ID
        emotion_id = item.get("emotion_id", 0)  # 0 = neutral

        return {
            "phonemes": torch.tensor(phoneme_ids, dtype=torch.long),
            "mel": torch.tensor(mel, dtype=torch.float32),
            "pitch": torch.tensor(pitch, dtype=torch.float32),
            "energy": torch.tensor(energy, dtype=torch.float32),
            "duration": torch.tensor(durations, dtype=torch.long),
            "emotion_id": torch.tensor(emotion_id, dtype=torch.long),
            "mel_len": torch.tensor(min_len, dtype=torch.long),
        }


def collate_tts_batch(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Collate TTS samples into padded batches.

    Args:
        batch: List of sample dicts from TTSDataset.

    Returns:
        Padded batch dict.
    """
    # Find max lengths
    max_phone_len = max(s["phonemes"].size(0) for s in batch)
    max_mel_len = max(s["mel"].size(0) for s in batch)
    n_mels = batch[0]["mel"].size(1)
    batch_size = len(batch)

    # Initialize padded tensors
    phonemes = torch.zeros(batch_size, max_phone_len, dtype=torch.long)
    mels = torch.zeros(batch_size, max_mel_len, n_mels)
    pitches = torch.zeros(batch_size, max_phone_len)
    energies = torch.zeros(batch_size, max_phone_len)
    durations = torch.zeros(batch_size, max_phone_len, dtype=torch.long)
    emotion_ids = torch.zeros(batch_size, dtype=torch.long)
    mel_lens = torch.zeros(batch_size, dtype=torch.long)

    for i, sample in enumerate(batch):
        plen = sample["phonemes"].size(0)
        mlen = sample["mel"].size(0)

        phonemes[i, :plen] = sample["phonemes"]
        mels[i, :mlen] = sample["mel"]
        durations[i, :plen] = sample["duration"]
        emotion_ids[i] = sample["emotion_id"]
        mel_lens[i] = sample["mel_len"]

        # Pitch and energy: map from mel frames to phoneme granularity
        # Average across frames assigned to each phoneme
        frame_idx = 0
        for j in range(plen):
            dur = sample["duration"][j].item()
            if dur > 0 and frame_idx + dur <= len(sample["pitch"]):
                pitches[i, j] = sample["pitch"][frame_idx:frame_idx + dur].mean()
                energies[i, j] = sample["energy"][frame_idx:frame_idx + dur].mean()
            frame_idx += dur

    return {
        "phonemes": phonemes,
        "mel": mels,
        "pitch": pitches,
        "energy": energies,
        "duration": durations,
        "emotion_id": emotion_ids,
        "mel_len": mel_lens,
    }


# ─── Manifest Builder ───────────────────────────────────────────────────────


EMOTION_LABEL_MAP = {
    "neutral": 0, "happy": 1, "sad": 2, "angry": 3,
    "surprise": 4, "fear": 5, "disgust": 6, "contempt": 7,
    "warm": 8,
}


def build_jenny_manifest(
    jenny_dir: str,
    output_path: str,
) -> None:
    """Build JSON manifest from Jenny dataset.

    Args:
        jenny_dir: Path to extracted Jenny dataset.
        output_path: Path to write manifest JSON.
    """
    manifest = []

    # Find metadata/transcript file
    meta_path = None
    for name in ["metadata.csv", "transcript.csv"]:
        candidate = os.path.join(jenny_dir, name)
        if os.path.exists(candidate):
            meta_path = candidate
            break

    if meta_path is None:
        # Walk directory for audio files
        print("[WARN] No metadata found, scanning for audio files...")
        for root, _, files in os.walk(jenny_dir):
            for f in sorted(files):
                if f.endswith((".wav", ".flac", ".mp3")):
                    manifest.append({
                        "audio_path": os.path.join(root, f),
                        "text": "",  # Will need manual transcripts
                        "emotion_id": 0,
                        "dataset": "jenny",
                    })
    else:
        with open(meta_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            for row in reader:
                if len(row) >= 2:
                    audio_id = row[0].strip()
                    text = row[-1].strip()
                    # Try common audio paths
                    audio_path = None
                    for ext in [".wav", ".flac", ".mp3"]:
                        for subdir in ["wavs", "wav", "audio", ""]:
                            candidate = os.path.join(
                                jenny_dir, subdir, f"{audio_id}{ext}"
                            )
                            if os.path.exists(candidate):
                                audio_path = candidate
                                break
                        if audio_path:
                            break

                    if audio_path and text:
                        manifest.append({
                            "audio_path": audio_path,
                            "text": text,
                            "emotion_id": 0,  # Jenny = neutral
                            "dataset": "jenny",
                        })

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Jenny manifest: {len(manifest)} samples → {output_path}")


def build_emns_manifest(
    emns_dir: str,
    output_path: str,
) -> None:
    """Build JSON manifest from EMNS dataset with emotion labels.

    Args:
        emns_dir: Path to extracted EMNS dataset.
        output_path: Path to write manifest JSON.
    """
    manifest = []
    meta_path = os.path.join(emns_dir, "metadata.csv")

    if not os.path.exists(meta_path):
        print(f"[ERROR] EMNS metadata not found at {meta_path}")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) >= 4:
                audio_file = row[0].strip()
                text = row[1].strip()
                emotion = row[3].strip().lower() if len(row) > 3 else "neutral"

                audio_path = os.path.join(emns_dir, audio_file)
                if not os.path.exists(audio_path):
                    # Try without directory prefix
                    audio_path = os.path.join(
                        emns_dir, os.path.basename(audio_file)
                    )

                emotion_id = EMOTION_LABEL_MAP.get(emotion, 0)

                manifest.append({
                    "audio_path": audio_path,
                    "text": text,
                    "emotion_id": emotion_id,
                    "emotion": emotion,
                    "dataset": "emns",
                })

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] EMNS manifest: {len(manifest)} samples → {output_path}")


def build_vctk_manifest(
    vctk_dir: str,
    output_path: str,
    speakers: Optional[List[str]] = None,
) -> None:
    """Build JSON manifest from VCTK British female speakers.

    Args:
        vctk_dir: Path to extracted VCTK dataset.
        output_path: Path to write manifest JSON.
        speakers: List of speaker IDs to include.
    """
    if speakers is None:
        speakers = VCTK_BRITISH_FEMALE_SPEAKERS

    manifest = []
    txt_dir = os.path.join(vctk_dir, "txt")
    wav_dir = os.path.join(vctk_dir, "wav48_silence_trimmed")

    # Handle nested extraction
    for subdir in ["VCTK-Corpus-0.92", ""]:
        test_txt = os.path.join(vctk_dir, subdir, "txt")
        if os.path.exists(test_txt):
            txt_dir = test_txt
            wav_dir = os.path.join(vctk_dir, subdir, "wav48_silence_trimmed")
            break

    for speaker in speakers:
        speaker_txt = os.path.join(txt_dir, speaker)
        speaker_wav = os.path.join(wav_dir, speaker)

        if not os.path.exists(speaker_txt):
            print(f"[WARN] VCTK speaker {speaker} not found, skipping")
            continue

        for txt_file in sorted(os.listdir(speaker_txt)):
            if not txt_file.endswith(".txt"):
                continue

            utt_id = txt_file.replace(".txt", "")
            txt_path = os.path.join(speaker_txt, txt_file)

            # VCTK uses .flac with _mic1/_mic2 suffixes
            audio_path = None
            for suffix in ["_mic1.flac", "_mic2.flac", ".flac", ".wav"]:
                candidate = os.path.join(speaker_wav, f"{utt_id}{suffix}")
                if os.path.exists(candidate):
                    audio_path = candidate
                    break

            if audio_path is None:
                continue

            with open(txt_path, "r") as f:
                text = f.read().strip()

            manifest.append({
                "audio_path": audio_path,
                "text": text,
                "emotion_id": 0,  # VCTK = neutral
                "dataset": "vctk",
                "speaker": speaker,
            })

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] VCTK manifest: {len(manifest)} samples → {output_path}")


def build_combined_manifest(
    data_dir: str = "data",
    output_path: str = "data/manifest_combined.json",
) -> str:
    """Build a combined manifest from all datasets.

    Args:
        data_dir: Root data directory.
        output_path: Path for combined manifest JSON.

    Returns:
        Path to the combined manifest.
    """
    combined = []

    # Jenny
    jenny_manifest = os.path.join(data_dir, "manifest_jenny.json")
    if os.path.exists(jenny_manifest):
        with open(jenny_manifest) as f:
            combined.extend(json.load(f))

    # EMNS
    emns_manifest = os.path.join(data_dir, "manifest_emns.json")
    if os.path.exists(emns_manifest):
        with open(emns_manifest) as f:
            combined.extend(json.load(f))

    # VCTK
    vctk_manifest = os.path.join(data_dir, "manifest_vctk.json")
    if os.path.exists(vctk_manifest):
        with open(vctk_manifest) as f:
            combined.extend(json.load(f))

    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"[INFO] Combined manifest: {len(combined)} total samples")
    return output_path


def create_dataloader(
    manifest_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    max_mel_len: int = 1000,
) -> DataLoader:
    """Create a TTS DataLoader from a manifest file.

    Args:
        manifest_path: Path to JSON manifest.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        shuffle: Whether to shuffle.
        max_mel_len: Maximum mel length.

    Returns:
        PyTorch DataLoader.
    """
    processor = AudioProcessor()
    frontend = TextFrontend()

    dataset = TTSDataset(
        manifest_path=manifest_path,
        audio_processor=processor,
        text_frontend=frontend,
        max_mel_len=max_mel_len,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_tts_batch,
        pin_memory=True,
        drop_last=True,
    )


# ─── CLI Entry Point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MambaTTS Data Pipeline")
    parser.add_argument(
        "--download", action="store_true",
        help="Download all datasets"
    )
    parser.add_argument(
        "--build-manifests", action="store_true",
        help="Build manifest JSON files"
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Root data directory"
    )
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    if args.download:
        print("=" * 60)
        print("  MambaTTS Data Pipeline — Downloading Datasets")
        print("=" * 60)
        jenny_dir = download_jenny(args.data_dir)
        emns_dir = download_emns(args.data_dir)
        vctk_dir = download_vctk(args.data_dir)
        print("\n[DONE] All datasets downloaded.")

    if args.build_manifests:
        print("=" * 60)
        print("  Building Training Manifests")
        print("=" * 60)
        build_jenny_manifest(
            os.path.join(args.data_dir, "jenny"),
            os.path.join(args.data_dir, "manifest_jenny.json"),
        )
        build_emns_manifest(
            os.path.join(args.data_dir, "emns"),
            os.path.join(args.data_dir, "manifest_emns.json"),
        )
        build_vctk_manifest(
            os.path.join(args.data_dir, "vctk"),
            os.path.join(args.data_dir, "manifest_vctk.json"),
        )
        build_combined_manifest(args.data_dir)
        print("\n[DONE] All manifests built.")
