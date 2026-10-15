"""
plot_results.py - يرسم asr_curve.png من results.csv بدون اعادة التقييم الطويل.

التشغيل (من مجلد المشروع):
  python plot_results.py
"""
import csv
import json
import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

series = OrderedDict()      # اسم الحالة -> قائمة (eps/255 ، ASR بالمئة)
n_valid = None

with open("results.csv", newline="") as f:
    for row in csv.DictReader(f):
        series.setdefault(row["condition"], []).append(
            (float(row["epsilon_255"]), 100 * float(row["asr"])))
        n_valid = int(row["n_valid"])

threshold = None
if os.path.exists("threshold.json"):
    with open("threshold.json") as f:
        threshold = json.load(f)["threshold"]

plt.figure(figsize=(9, 5.5))
for name, pts in series.items():
    pts.sort()
    xs, ys = zip(*pts)
    style = "--" if "non-adaptive" in name else "-"
    plt.plot(xs, ys, marker="o", linestyle=style, label=name)

title = f"ASR on LFW (n={n_valid}"
if threshold is not None:
    title += f", threshold={threshold:.3f}"
title += ")"

plt.xlabel("Perturbation budget ε (≈ x/255 pixel scale)")
plt.ylabel("Attack Success Rate (%)")
plt.title(title)
plt.ylim(-2, 102)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("asr_curve.png", dpi=250)
print("انحفظت: asr_curve.png")