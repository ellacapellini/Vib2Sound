"""
training script Vib2Sound + CMC Loss
only + to the training loop:
- subclasses Vib2Sound_multichannel to expose presigmoid FC2 output needed for AVCBlock 
- resues vib2sound create_Dataloader, audio. hparams, writer exaclty as original trainer.py does
- checkpoint saves vib2sound weights separately so they stay compatible with original inference.py

"""

import os
import sys
import math
import argparse
import logging
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F

# point python at vib2sound-lab
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB_PATH   = os.path.join(REPO_ROOT, "vib2sound-lab")
ELLA_PATH  = os.path.dirname(os.path.abspath(__file__))

for p in [LAB_PATH, ELLA_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

# vib2sound imports untouched
from utils.hparams import HParam
from utils.audio import Audio
from utils.writer import MyWriter
from dataloader import create_dataloader
from model.model_multichannel import Vib2Sound_multichannel

#CMC modules 
from cmc import AVCBlock, AccelEmbedder, CMCLoss, check_temporal_alignment

#subclass: expose presimgoid FC2 output for CMC (no lab code changes)

class Vib2Sound_multichannel_CMC(Vib2Sound_multichannel):
    """
    = to Vib2Sound_multichannel, but forward() additionally returns the pre-sigmoid FC2 outputs (raw_target, raw_nontarget) needed by AVCBlock.
    added lines are marked with  #!! 
    Everything else is a verbatim copy 
    """
    def forward(self, x_mixture, x_phase, x_radio1, x_radio2):
        channel_num = self.hp.train.channel_num
        x_mixture = x_mixture[:, 0:channel_num, :, :]
        x_phase   = x_phase[:, 0:channel_num, :, :]
        x_combined = torch.cat([x_mixture, x_phase], dim=1)
        x_combined = self.conv(x_combined)
        x_combined = x_combined.transpose(1, 2).contiguous()
        x_combined = x_combined.view(x_combined.size(0), x_combined.size(1), -1)

        #target bach 
        x_target = torch.cat((x_combined, x_radio1), dim=2)
        x_target, _ = self.lstm(x_target)
        x_target = F.relu(x_target)
        x_target = self.fc1(x_target)
        x_target = F.relu(x_target)
        raw_target = self.fc2(x_target)         #!!
        mask_target = torch.sigmoid(raw_target)

        #nontarget bach 
        x_nontarget = torch.cat((x_combined, x_radio2), dim=2)
        x_nontarget, _ = self.lstm(x_nontarget)
        x_nontarget = F.relu(x_nontarget)
        x_nontarget = self.fc1(x_nontarget)
        x_nontarget = F.relu(x_nontarget)
        raw_nontarget = self.fc2(x_nontarget)    #!!
        mask_nontarget = torch.sigmoid(raw_nontarget)
        return mask_target, mask_nontarget, raw_target, raw_nontarget


#training
def train_cmc(args, hp, hp_str):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    #data
    trainloader = create_dataloader(hp, args, "train")
    testloader  = create_dataloader(hp, args, "test")

    #models
    model = Vib2Sound_multichannel_CMC(hp).cuda()
    n_freq    = hp.audio.num_freq          #193
    embed_dim = args.embed_dim             #default 128
    avc_block      = AVCBlock(n_freq=n_freq, embed_dim=embed_dim).cuda()
    accel_embedder = AccelEmbedder(n_freq=n_freq, embed_dim=embed_dim).cuda()
    cmc_loss_fn    = CMCLoss()

    # optimiser: one Adam covering all trainable params 
    optimizer = torch.optim.Adam(
        list(model.parameters()) +
        list(avc_block.parameters()) +
        list(accel_embedder.parameters()),
        lr=hp.train.adam,
    )

    # resume from checkpoint if gvien 
    step = 0
    if args.chkpt_path is not None:
        logger.info(f"Resuming from checkpoint: {args.chkpt_path}")
        ckpt = torch.load(args.chkpt_path)
        model.load_state_dict(ckpt["model"])
        if "avc_block" in ckpt:
            avc_block.load_state_dict(ckpt["avc_block"])
            accel_embedder.load_state_dict(ckpt["accel_embedder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        logger.info(f"Resumed at step {step}")
    else:
        logger.info("Starting new CMC training run")

    #logging
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = MyWriter(hp, args.log_dir)

    #training loop
    criterion = nn.MSELoss()
    first_batch = True

    try:
        while True:
            model.train()
            avc_block.train()
            accel_embedder.train()
            for (radio1_mag, radio2_mag, target_mag, nontarget_mag,
                 mixed_mag, mixed_phase, target_wav, nontarget_wav) in trainloader:
                #move to GPU and cast (=original train.py) 
                radio1_mag    = radio1_mag.cuda().float() #(B, T, 193)
                radio2_mag    = radio2_mag.cuda().float() #(B, T, 193)
                target_mag    = target_mag.cuda().float() #(B, T, 193)
                nontarget_mag = nontarget_mag.cuda().float() #(B, T, 193)
                mixed_mag     = mixed_mag.cuda().float() #(B, n_mics, T, 193)
                mixed_phase   = mixed_phase.cuda().float() #(B, n_mics, T, 193)

                # vib2sound forward
                mask_target, mask_nontarget, raw_target, raw_nontarget = model(
                    mixed_mag, mixed_phase, radio1_mag, radio2_mag
                )
                #all four: (B, T, 193)

                #MSE loss (= original train.py)
                #apply mask to first mic channel: mixed_mag[:, 0, :, :] = (B, T, 193)
                mixed_mag_ch0 = mixed_mag[:, 0, :, :] # (B, T, 193)
                output_target    = mixed_mag_ch0 * mask_target
                output_nontarget = mixed_mag_ch0 * mask_nontarget
                loss_mse = (criterion(output_target,    target_mag) +
                            criterion(output_nontarget, nontarget_mag))

                #!! CMC loss
                #accelr. embed. —> input already (B, T, 193)
                cv1 = accel_embedder(radio1_mag) #(B, T, embed_dim)
                cv2 = accel_embedder(radio2_mag)#(B, T, embed_dim)

                #audio embed. —> from pre-sigmoid FC2 output (B, T, 193)
                ca1 = avc_block(raw_target) #(B, T, embed_dim)
                ca2 = avc_block(raw_nontarget) #(B, T, embed_dim)

                #shape check on first batch only
                if first_batch:
                    check_temporal_alignment(cv1, ca1, "bird1")
                    check_temporal_alignment(cv2, ca2, "bird2")
                    logger.info(f"Shape check passed. "
                                f"accel_embed: {cv1.shape}, audio_embed: {ca1.shape}")
                    first_batch = False
                cmc_out = cmc_loss_fn(
                    accel_embeds=[cv1, cv2],
                    audio_embeds=[ca1, ca2],
                )
                loss = loss_mse + args.lambda_cmc * cmc_out.loss
                
                #backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                loss_val = loss.item()
                if loss_val > 1e8 or math.isnan(loss_val):
                    logger.error(f"Loss exploded to {loss_val:.2f} at step {step}")
                    raise Exception("Loss exploded")
                #logging
                if step % hp.train.summary_interval == 0:
                    writer.log_training(loss_val, step)
                    #log CMC diagnostics alongside MSE
                    writer.writer.add_scalar("loss/mse",     loss_mse.item(),       step)
                    writer.writer.add_scalar("loss/cmc",     cmc_out.loss.item(),   step)
                    writer.writer.add_scalar("cmc/cos_pos",  cmc_out.cos_pos.item(), step)
                    writer.writer.add_scalar("cmc/cos_neg",  cmc_out.cos_neg.item(), step)
                    logger.info(
                        f"step {step:>7d} | "
                        f"total {loss_val:.4f} | "
                        f"mse {loss_mse.item():.4f} | "
                        f"cmc {cmc_out.loss.item():.4f} | "
                        f"cos+ {cmc_out.cos_pos.item():.3f} | "
                        f"cos- {cmc_out.cos_neg.item():.3f}"
                    )
                #checkpoint
                if step % hp.train.checkpoint_interval == 0:
                    save_path = os.path.join(
                        args.checkpoint_dir, f"chkpt_{step}.pt"
                    )
                    torch.save({
                        "model":          model.state_dict(),   #compatible with original inference.py
                        "avc_block":      avc_block.state_dict(),
                        "accel_embedder": accel_embedder.state_dict(),
                        "optimizer":      optimizer.state_dict(),
                        "step":           step,
                        "hp_str":         hp_str,
                        "lambda_cmc":     args.lambda_cmc,
                    }, save_path)
                    logger.info(f"Saved checkpoint: {save_path}")
                if args.max_steps is not None and step >= args.max_steps:
                    logger.info(f"Reached max steps ({args.max_steps}). Done.")
                    return
    except Exception as e:
        logger.error(f"Exiting due to exception: {e}")
        traceback.print_exc()


#entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="vib2sound-lab/config/config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--num_channels", type=str, default="multi",
                        choices=["single", "multi"])
    parser.add_argument("--lambda_cmc", type=float, default=1.0,
                        help="CMC loss weight. Start at 1.0, sweep {0.1, 1.0, 10.0}")
    parser.add_argument("--embed_dim", type=int, default=128,
                        help="Dimension of CMC embedding space")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="checkpoints/cmc")
    parser.add_argument("--log_dir", type=str,
                        default="logs/cmc")
    parser.add_argument("--chkpt_path", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, "r") as f:
        hp_str = f.read()

    if not os.path.isabs(hp.data.clean_data_dir):
        hp.data.clean_data_dir = os.path.join(LAB_PATH, hp.data.clean_data_dir)
    print(f"Data directory: {hp.data.clean_data_dir}")

    train_cmc(args, hp, hp_str)