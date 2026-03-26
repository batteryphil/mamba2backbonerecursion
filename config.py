"""MambaTTS configuration — all hyperparameters in one place."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AudioConfig:
    """Audio processing parameters."""

    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0
    ref_db: float = 20.0
    max_db: float = 100.0
    griffin_lim_iters: int = 60


@dataclass
class MambaConfig:
    """Mamba SSM block parameters."""

    d_model: int = 256
    d_state: int = 16
    d_conv: int = 4
    expand_factor: int = 2
    dt_rank: str = "auto"  # "auto" = d_model // 16
    bias: bool = False
    conv_bias: bool = True


@dataclass
class ModelConfig:
    """MambaTTS model architecture."""

    # Phoneme vocabulary
    phoneme_vocab_size: int = 85
    phoneme_pad_id: int = 0

    # Encoder
    encoder_layers: int = 6
    encoder_d_model: int = 256

    # Decoder
    decoder_layers: int = 6
    decoder_d_model: int = 256

    # Emotion conditioning
    num_emotions: int = 9  # 8 + neutral
    emotion_embed_dim: int = 256

    # Duration predictor
    dur_predictor_channels: int = 256
    dur_predictor_kernel: int = 3
    dur_predictor_layers: int = 2
    dur_predictor_dropout: float = 0.1

    # Pitch predictor
    pitch_predictor_channels: int = 256
    pitch_predictor_kernel: int = 3
    pitch_predictor_layers: int = 2
    pitch_predictor_dropout: float = 0.1
    pitch_embed_dim: int = 256

    # Energy predictor
    energy_predictor_channels: int = 256
    energy_predictor_kernel: int = 3
    energy_predictor_layers: int = 2
    energy_predictor_dropout: float = 0.1
    energy_embed_dim: int = 256

    # PostNet
    postnet_channels: int = 256
    postnet_kernel: int = 5
    postnet_layers: int = 5

    # Output
    n_mels: int = 80

    # Mamba block config
    mamba: MambaConfig = field(default_factory=MambaConfig)


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Two-stage training
    stage: int = 1  # 1 = base voice (Jenny), 2 = emotion fine-tune (EMNS)

    # Optimizer
    lr: float = 1e-3
    lr_finetune: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.98)
    grad_clip: float = 1.0

    # Schedule
    warmup_steps: int = 4000

    # Training
    epochs: int = 200
    batch_size: int = 16
    num_workers: int = 4
    seed: int = 42

    # Logging
    log_every: int = 50
    checkpoint_every: int = 200
    eval_every: int = 1000

    # Loss weights
    mel_loss_weight: float = 1.0
    postnet_mel_loss_weight: float = 1.0
    duration_loss_weight: float = 1.0
    pitch_loss_weight: float = 0.5
    energy_loss_weight: float = 0.5

    # Paths
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class InferenceConfig:
    """Inference settings."""

    device: str = "cpu"
    emotion: str = "neutral"
    speed: float = 1.0
    pitch_shift: float = 0.0
    energy_scale: float = 1.0


# Emotion label mapping
EMOTION_MAP = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "surprise": 4,
    "fear": 5,
    "disgust": 6,
    "contempt": 7,
    "warm": 8,  # Alias for blend of happy + neutral
}
