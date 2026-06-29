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

def parse(text: str) -> list[dict]:
    rows = []
    for m in PATTERN.finditer(text):
        rows.append({
            "timestamp":  m.group(1),
            "step":       int(m.group(2)),
            "loss_total": float(m.group(3)),
            "loss_mse":   float(m.group(4)),
            "loss_cmc":   float(m.group(5)),
            "cos_pos":    float(m.group(6)),
            "cos_neg":    float(m.group(7)),
        })
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Path to log file. Reads stdin if omitted.")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file.")
    