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
    """
    projects FC layer output (soft mask, pre-sigmoid) into shared embedding space where CMC loss operates
    input: (B,T, n_freq)
    output: (B, T, embed_dim)
    this modules is shared across birds (following Makishima)= one instance handles both bird 1 nd 2
    """
    def __init__(self, n_freq: int = 193, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_freq, embed_dim),
            nn.Tanh(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) #(B, T, embed_dim)

class AccelEmbedder(nn.Module):
    """
    encodes accelr. magnitued spectro. into the same embedding space as AVCBlock
    Input: (B, 1, n_freq_a, T)
    output: (B, T, embed_dim)
    """
    def __init__(self, n_freq_q: int = 193, embed_dim: int = 128): 
        super().__init__()
        self.conv = nn.Sequential(
            #(B, 1, F, T) -> (B, 32, F, T)
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            #(B, 32, F, T) -> (B, 64, F, T)
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=(1,1)),
            nn.ReLu(),
            #collapse freq. axis (B, 64, 1, T)
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.proj = nn.Linear(64, embed_dim)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        h = self.conv(x) #(B, 64, 1, T)
        h = h.squeeze(2).permute(0, 2, 1) #(B, T, 64)
        return self.proj(h) #(B, T, embed_dim)
    
def check_temporal_alignment(
    accel_embed: torch.Tensor,
    audio_embed: torch.Tensor,
    label: str = "",
) -> None:
    """
    hard check both embed share the same (B, T, D)
    calling it in the first training batch and then silent after.
    should raise ValueError with a descriptive message if shape don't match
    """
    if accel_embed.shape != audio_embed.shape:
        raise ValueError(
            f"[CMC{' ' + label if label else ''}] Shape mismatch:\n"
            f"accel_embed: {tuple(accel_embed.shape)}\n"
            f" audio_embed: {tuple(audio_embed.shape)}\n"
            f"The time (T) and embed_dim (D) dimensions must match.\n"
            f"Check that AccelEmbedder and AVCBlock use the same embed_dim,\n"
            f"and that the STFT hop size produces the same T for both streams."
        )

    
    
        