"""
converts terminal output from train_cmc.py into a CSV training log
"""
import re
import csv
import sys
import argparse
from pathlib import Path
from datetime import datetime

PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})" #timestamp
    r"[,\d]*\s+INFO\s+"
    r"step\s+(\d+)\s*\|" #step
    r"\s*total\s+([\-\d.]+)\s*\|" #loss_total
    r"\s*mse\s+([\-\d.]+)\s*\|" #loss_mse
    r"\s*cmc\s+([\-\d.]+)\s*\|" #loss_cmc
    r"\s*cos\+\s*([\-\d.]+)\s*\|" #cos_pos
    r"\s*cos-\s*([\-\d.]+)" #cos_neg
)
FIELDS = ["timestamp", "step", "loss_total", "loss_mse", "loss_cmc", "cos_pos", "cos_neg"]


