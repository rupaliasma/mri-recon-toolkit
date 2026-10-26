"""Turn results/*/summary.json into a Markdown table."""

import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
rows = []
for path in sorted(root.glob("*/summary.json")):
    summary = json.loads(path.read_text())
    setting = f"{summary['data']['mask']} x{summary['data']['accel']:g}"
    for method, stats in summary["methods"].items():
        rows.append((setting, method, stats))

seen = set()
print("| Sampling | Method | SSIM | PSNR (dB) | NMSE | Slices |")
print("|---|---|---|---|---|---|")
for setting, method, s in rows:
    if method == "zero-filled":
        if setting in seen:
            continue
        seen.add(setting)
    print(
        f"| {setting} | {method} | {s['ssim']['mean']:.4f} ± {s['ssim']['std']:.4f} "
        f"| {s['psnr']['mean']:.2f} ± {s['psnr']['std']:.2f} | {s['nmse']['mean']:.4f} | {s['n_slices']} |"
    )
