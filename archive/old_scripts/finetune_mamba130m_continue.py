"""
finetune_mamba130m_continue.py — Resume fine-tune from Step 500 checkpoint.
Runs 2000 more steps with improved answer-termination format.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel
import json, random, time, os

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
N_REASON   = 3
STEPS      = 2000
LR         = 5e-5      # Lower than initial run — already converging
SEQ_LEN    = 256
BATCH_SIZE = 4
ACCUM      = 8
LOG_EVERY  = 25
CKPT_IN    = "mamba130m_finetuned.pt"
CKPT_OUT   = "mamba130m_finetuned_v2.pt"

EOS = "
