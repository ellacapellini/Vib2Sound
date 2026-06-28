"""
Audio-Visual Correspondence (AVC) projection heads for CMC loss
for TRAINING-ONLY.
These modules are instantiated alongside the Vib2Sound model, used to compute CMCLoss and discarded at inference.
Architecture background (from Akahoshi et al. Fig. 2):
    - CNN (8 layers) processes mixture microphone spectograms
    - Accelr. magnitude spectog. is conctatenated to CNN output 
    - LSTM (1 layer) processes concatenation
    - FC (2 LAYERS) prdicts soft masks <- tap here for AVCBlock
    - soft masks applied to mic1 magnitude -> iSTFT
CMC loss (Makishima eq.10) requires: 
    - accel_embed : (B, T, D) from AccelEmbedder(accel_spectogram)
    - audio_embed : (B, T, D) from AVCBlock(fc_output/pre_sigmoid mask)
STFT param:
FFT: 384 samples, hop: 96 samples, sr: 244414 Hz
    - frequency bins (one-sided): 193
    - acccel. is 200Hz high-pass filtered, same STFT setting    
"""
import torch
import torch.nn as nn

class AVCBlock(nn.Module):
    