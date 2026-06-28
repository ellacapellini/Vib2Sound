"""
Audio-Visual Correspondence (AVC) projection heads for CMC loss
for TRAINING-ONLY (discardd at inference)
i tried to match it exaclty to Vib2Sound_multichannel tensor shapes (from model_multichannel.py and config.yaml)
    radio1_mag/radio2_mag : (B, T, 93) <- already (B, T, F) no permute needed
    fc2 output (presigmoid): (B, T, 193) <- same shape, will be my AVCBlock input
    n_fft = 384
    num_freq = 193 (n_fft//2+1)
    lstm_dim = 400 -> LSTM output = 2*400= 800 (bidirectional)
    fc1_dm = 600
    fc2_dm = 193 
AVCBlock: (B, T, 193) -> (B, T, embed_dim) project presigmoid FC2 output 
AccelEmbedder: (B, T, 193) -> (B, T, embed_dim) projects radio mag spectrogram 
    
"""
import torch
import torch.nn as nn

class AVCBlock(nn.Module):
    """
    projects presigmoid FC2 output into the CMC embedding space.
    Input:  (B, T, 193) —> raw FC2 output, before torch.sigmoid()
    Output: (B, T, embed_dim)
    shared across target and nontarget birds (Makishima et al. share weights
    across speakers; so i think it's appropriate since both birds are same species).
    """

    def __init__(self, n_freq: int = 193, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_freq, embed_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class AccelEmbedder(nn.Module):
    """
    projects accelr. magnitude spectr. into the same CMC embedding space as AVCBlock.
    In Vib2Sound, radio_mag arrives from dataloader as (B, T, 193).
    kept lightweight —> the accelr. is the reference anchor, not a rich encoder.
    Input:  (B, T, 193)
    Output: (B, T, embed_dim)
    """

    def __init__(self, n_freq: int = 193, embed_dim: int = 128, hidden: int = 64):
        super().__init__()
        #project frequency axis first
        self.freq_proj = nn.Linear(n_freq, hidden)
        #small temporal conv to capture shortrange dynamics
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),      
        )
        self.out_proj = nn.Linear(hidden, embed_dim)  
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.freq_proj(x)) #(B, T, hidden)
        h = h.permute(0, 2, 1) #(B, hidden, T) for Conv1d
        h = self.temporal(h) #(B, hidden, T)
        h = h.permute(0, 2, 1) #(B, T, hidden)
        return self.out_proj(h) #(B, T, embed_dim)

def check_temporal_alignment(
    accel_embed: torch.Tensor,
    audio_embed: torch.Tensor,
    label: str = "",
) -> None:
    """
    Hhardard check that accel and audio embed have = shape.
    call once on the first training batch; raises ValueError if mismatch.
    """
    if accel_embed.shape != audio_embed.shape:
        raise ValueError(
            f"[CMC shape mismatch{' — ' + label if label else ''}]\n"
            f"  accel_embed : {tuple(accel_embed.shape)}\n"
            f"  audio_embed : {tuple(audio_embed.shape)}\n"
            f"Both must be (B, T, embed_dim). "
            f"Check embed_dim matches in AVCBlock and AccelEmbedder."
        )