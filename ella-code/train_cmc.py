"""
training script Vib2Sound + CMC Loss
only + to the training loop:
1. 2 extra modules (AVCBlokc, AccelEmbedder) for training onluy
2. CMCLoss computed alongisde existing MSE Loss
3. Additional logging of cos_pos / cos_neg diagnosticis 
"""
import os 
import sys 
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

