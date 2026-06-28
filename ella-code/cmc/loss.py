"""
Cross-Modal Correspondence (CMC) loss - Makishima et al. eq. 10
For N birds, at each time fram j:
   - MAXIMISE cosine similarity between matched (accel_n, audio_n) pairs
   - MINIMISE cosine similarity between mismatched (Accel_n, audio_n') pairs
With N=2 birds (BirdPark / Vib2Sound settings), each bird has exactly one positive pair and one negative pair x frame.
Total training loss:
    L = L_MSE + lambda_cmc * L_CMC
Makishima found lambda_cmc = 1.0 optimal on LRS3-TED
For Vib2Sound, I'll start at 1.0 and sweep {0.1, 1.0, 10.0}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional 

@dataclass
class CMCLossOutput:
    """
    Structured output from CMCLoss.forward()
       loss = total CMC loss (scalar) plug into L_total = L_MSE + λ * loss
       cos_pos = mean cosine siilarity of matched pairs (want → +1 over training)
       cos_neg = mean cosine similarity of mimatched pairs (Want → 0 over training)
    Log cos_pos and cos_neg to WandB/TesnorBoard from day one
    Divergence of these 2 curves is my main diagnostic
    """
    loss: torch.Tensor
    cos_pos: torch.Tensor
    cos_neg: torch.Tensor

class CMCLoss(nn.Module):
    """
    CMC loss for N birds
    usage:
       cmc_loss_fn = CMCLoss()
       out = cmc_loss_fn(
           accel_embeds=[cv_1, cv_2], #lit of (B, T, D)
           audio_embeds=[ca_1, ca_2], #list of (B, T, D)
       )
       total_loss = mse_loss + lambda_cmc * out.loss
       wandb.log({"cmc_loss": out.cos_pos, "cmc/neg": out.cos_neg})
    """
    def forward(
        self,
        accel_embeds: list[torch.Tensor],
        audio_embeds: list[torch.Tensor],
    ) -> CMCLossOutput:
        """args:
        accel_embdes = list of n tensors, each (B, T, D) outputs of accelEmbedder for each bird
        audio_embeds: list of N tesnosrs, each (B, T, D) outputs of AVCBlock for each bird
        rturns:
        CMCLossOutput with .loss, .cos_pos, .cos_neg
        """
        N = len(accel_embeds)
        assert N == len(audio_embeds), "must provide = n of accel + audio embeddings"
        assert N >= 2, "CMC loss requires at least 2 birds"
        device = accel_embeds[0].device
        total_loss = torch.tensor(0.0, device=device)
        pos_sims: list[torch.Tensor] = []
        neg_sims: list[torch.Tensor] = []
        for n in range(N):
            cv = accel_embeds[n] #(B, T, D) accel embedding  bird n 
            ca = audio_embeds[n] #(B, T, D) audio embedding bird n 
            # positive = same bird, cosine similairyt i want to MAXIMISE
            # cosine_similarity returns (B, T); take mean across batch and time 
            pos = F.cosine_similarity(cv, ca, dim=1).mean() # scalar
            pos_sims.append(pos)
            # negative = other birds, cosine similairity i wnat to MINIMISE
            for n_prime in range(N):
                if n_prime == n:
                    continue
                ca_prime = audio_embeds[n_prime] #(B, T, D)
                neg = F.cosine_similarity(cv, ca_prime, dim=-1).mean()
                neg_sims.append(neg)
                # contribution to loss for bird n = push neg up (+), pull pos down (-)
                neg_sum = sum(
                    F.cosine_similarity(cv, audio_embeds[np_], dim=-1).mean()
                    for np_ in range(N) if np_ != n
                )
                total_loss = total_loss + (neg_sum - pos)
        mean_pos = torch.stack(pos_sims).mean()
        mean_neg = torch.stack(neg_sims).mean()
        
        return CMCLossOutput(
            loss=total_loss,
            cos_pos=mean_pos,
            cos_neg=mean_neg,
        )