"""
التقييم العلمي: نسبة نجاح الهجوم (ASR) على مئات الأزواج من LFW.

لكل زوج "نفس الشخص" (ويقبله النظام بدون هجوم) نطبق:
  FGSM, PGD, PGD ثم Blur/JPEG (المهاجم ما يعرف بالدفاع),
  وهجوم تكيّفي ضد Blur و JPEG (المهاجم يعرف بالدفاع).
ASR = نسبة الأزواج اللي صار النظام يرفضها بعد الهجوم.

التشغيل (بعد lfw_eval.py):
  python lfw_attack_eval.py --n 100
  python lfw_attack_eval.py --n 300 --eps 0.01,0.03,0.05,0.08

ينتج: results.csv و asr_curve.png
"""
import argparse
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.datasets import fetch_lfw_pairs
from tabulate import tabulate

import attacks as A

CONDITIONS = [
    "FGSM",
    "PGD",
    "PGD -> Blur (non-adaptive)",
    "Adaptive vs Blur",
    "PGD -> JPEG (non-adaptive)",
    "Adaptive vs JPEG",
]


def to_pil(arr):
    return Image.fromarray((arr * 255).astype(np.uint8))


def wilson(k, n, z=1.96):
    """فاصل ثقة 95% لنسبة (Wilson interval)"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def evaluate_pair(face1, ref, eps, steps):
    """يرجع التشابه النهائي لكل حالة"""
    out = {}
    out["FGSM"] = A.similarity(A.fgsm(face1, ref, eps), ref)

    pgd_face = A.pgd(face1, ref, eps, steps=steps)
    out["PGD"] = A.similarity(pgd_face, ref)
    out["PGD -> Blur (non-adaptive)"] = A.similarity(A.blur_defense(pgd_face), ref)
    out["PGD -> JPEG (non-adaptive)"] = A.similarity(A.jpeg_defense(pgd_face), ref)

    ad_blur = A.pgd(face1, ref, eps, steps=steps, preprocess=A.blur_defense)
    out["Adaptive vs Blur"] = A.similarity(A.blur_defense(ad_blur), ref)

    ad_jpeg = A.pgd(face1, ref, eps, steps=steps, preprocess=A.jpeg_bpda)
    out["Adaptive vs JPEG"] = A.similarity(A.jpeg_defense(ad_jpeg), ref)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="عدد الأزواج الصالحة")
    ap.add_argument("--eps", type=str, default="0.01,0.03,0.05,0.08")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    eps_list = [float(e) for e in args.eps.split(",")]

    th, calibrated = A.load_threshold()
    if not calibrated:
        print("⚠ threshold.json غير موجود! شغّل lfw_eval.py اولًا (نستخدم 0.5 مؤقتًا).")
    print(f"Threshold = {th:.3f} | Device = {A.device}")

    data = fetch_lfw_pairs(subset="test", color=True, resize=1.0,
                           slice_=(slice(0, 250), slice(0, 250)))
    pairs, target = data.pairs, data.target
    rng = np.random.default_rng(args.seed)
    pos_idx = rng.permutation(np.where(target == 1)[0])   # ازواج نفس الشخص فقط

    success = {c: {e: 0 for e in eps_list} for c in CONDITIONS}
    sim_sum = {c: {e: 0.0 for e in eps_list} for c in CONDITIONS}
    valid = skipped_noface = skipped_rejected = 0
    clean_blur_ok = clean_jpeg_ok = 0
    t0 = time.time()

    for i in pos_idx:
        if valid >= args.n:
            break
        f1 = A.face_from_pil(to_pil(pairs[i][0]))
        f2 = A.face_from_pil(to_pil(pairs[i][1]))
        if f1 is None or f2 is None:
            skipped_noface += 1
            continue
        ref = A.reference_embedding(f2)
        if A.similarity(f1, ref) < th:        # النظام يرفض الزوج اصلًا، ما نحسبه
            skipped_rejected += 1
            continue

        valid += 1
        clean_blur_ok += int(A.similarity(A.blur_defense(f1), ref) >= th)
        clean_jpeg_ok += int(A.similarity(A.jpeg_defense(f1), ref) >= th)

        for eps in eps_list:
            sims = evaluate_pair(f1, ref, eps, args.steps)
            for c, s in sims.items():
                sim_sum[c][eps] += s
                success[c][eps] += int(s < th)

        if valid % 10 == 0:
            el = time.time() - t0
            print(f"  {valid}/{args.n} ازواج | {el / 60:.1f} دقيقة | "
                  f"المتبقي ≈ {el / valid * (args.n - valid) / 60:.1f} دقيقة")

    if valid == 0:
        raise SystemExit("ما اكو ازواج صالحة. تأكد من threshold.json.")

    # ---------------- النتايج ----------------
    print(f"\nازواج صالحة: {valid} | بدون وجه: {skipped_noface} | مرفوضة اصلًا بدون هجوم: {skipped_rejected}")
    print(f"القبول النظيف بعد Blur: {clean_blur_ok / valid:.1%} | بعد JPEG: {clean_jpeg_ok / valid:.1%}"
          "  (هذي كلفة الدفاع على الاستخدام الطبيعي)\n")

    header = ["Condition"] + [f"eps={e:g} (~{A.eps_to_255(e):.1f}/255)" for e in eps_list]
    table, csv_rows = [], []
    for c in CONDITIONS:
        row = [c]
        for e in eps_list:
            k = success[c][e]
            lo, hi = wilson(k, valid)
            row.append(f"{k / valid:.0%} ({lo:.0%}-{hi:.0%})")
            csv_rows.append([c, e, round(A.eps_to_255(e), 2), valid, k,
                             round(k / valid, 4), round(lo, 4), round(hi, 4),
                             round(sim_sum[c][e] / valid, 4)])
        table.append(row)
    print("ASR = نسبة نجاح الهجوم، بين القوسين فاصل ثقة 95%")
    print(tabulate(table, headers=header, tablefmt="fancy_grid"))

    with open("results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "epsilon", "epsilon_255", "n_valid", "successes",
                    "asr", "ci_low", "ci_high", "mean_similarity"])
        w.writerows(csv_rows)

    plt.figure(figsize=(9, 5.5))
    for c in CONDITIONS:
        ys = [100 * success[c][e] / valid for e in eps_list]
        style = "--" if "non-adaptive" in c else "-"
        plt.plot([A.eps_to_255(e) for e in eps_list], ys, marker="o", linestyle=style, label=c)
    plt.xlabel("Perturbation budget ε (≈ x/255 pixel scale)")
    plt.ylabel("Attack Success Rate (%)")
    plt.title(f"ASR on LFW (n={valid}, threshold={th:.3f})")
    plt.ylim(-2, 102)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("asr_curve.png", dpi=250)
    print("\nانحفظت: results.csv و asr_curve.png")


if __name__ == "__main__":
    main()