#!/usr/bin/env python3
from __future__ import annotations
import os, json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"

def main():
    leaderboard_fp = REPORTS / "leaderboard.json"
    if not leaderboard_fp.exists():
        raise SystemExit("leaderboard.json not found. Run evaluation first (python src/evaluate.py).")

    leaderboard = json.loads(leaderboard_fp.read_text())

    lines = []
    lines.append("# Breast Cancer Prediction — Results Summary")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_\n")

    # Leaderboard
    lines.append("## Leaderboard (thresholded for recall target)")
    lines.append("| Model | Threshold | Recall | Precision | F1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in leaderboard:
        lines.append(f"| {row['model']} | {float(row['threshold']):.6f} | {float(row['recall']):.4f} | {float(row['precision']):.4f} | {float(row['f1']):.4f} |")

    # Per-model details
    lines.append("\n## Per-Model Evaluation")
    for row in leaderboard:
        name = row["model"]
        eval_fp = REPORTS / f"eval_{name}.json"
        if not eval_fp.exists():
            continue
        ev = json.loads(eval_fp.read_text())
        lines.append(f"### {name}")
        lines.append(f"- Threshold: **{float(ev['threshold']):.6f}**  ")
        lines.append(f"- Recall: **{float(ev['recall']):.4f}**, Precision: **{float(ev['precision']):.4f}**, F1: **{float(ev['f1']):.4f}**  ")
        cm = ev.get("confusion_matrix", [[0,0],[0,0]])
        lines.append(f"- Confusion Matrix (TN, FP / FN, TP): `{cm}`  ")

        roc_png = FIG / f"roc_{name}.png"
        pr_png  = FIG / f"pr_{name}.png"
        if roc_png.exists():
            lines.append(f"![ROC {name}](figures/{roc_png.name})")
        if pr_png.exists():
            lines.append(f"![PR {name}](figures/{pr_png.name})")
        lines.append("")

    out_md = REPORTS / "final_report.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}")

if __name__ == "__main__":
    main()
