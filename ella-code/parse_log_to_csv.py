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
    parser.add_argument("--append", action="store_true", help="Append to existing CSV instead of overwriting.")
    args = parser.parse_args()
    #read input 
    if args.input:
        text = Path(args.input).read_text()
    else:
        print("Paste log output below, then press CTRL+D (Mac/Linux) or Ctrl+Z (Windows):")
        text = sys.stdin.read()
    rows = parse(text)
    if not rows:
        print("No matching log lines found. Check log format")
        sys.exit(1)
    #write CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append and out_path.exists() else "n"
    write_header = mode == "w" or not out_path.exists()
    with open (out_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    
    print(f"Parsed {len(rows)} steps → {out_path}")
    print(f"  Steps: {rows[0]['step']} – {rows[-1]['step']}")
    print(f"  cos_pos range: {min(r['cos_pos'] for r in rows):.3f} – {max(r['cos_pos'] for r in rows):.3f}")
    print(f"  cos_neg range: {min(r['cos_neg'] for r in rows):.3f} – {max(r['cos_neg'] for r in rows):.3f}")
    print(f"  mse range:     {min(r['loss_mse'] for r in rows):.4f} – {max(r['loss_mse'] for r in rows):.4f}")
 

if __name__ == "__main__":
    main()