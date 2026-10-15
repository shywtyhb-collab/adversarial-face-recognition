"""
الاسبوع 2: دقة النظام النظيفة على LFW ومعايرة الـ threshold.

يستخدم نفس النموذج وكاشف الوجه من attacks.py حتى يكون الـ threshold
مناسب 100% للهجمات (نفس معالجة الصور بالضبط).

التشغيل:
  python lfw_eval.py --n 500
ينتج: threshold.json و roc_curve.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import auc, roc_curve

import attacks as A

torch.manual_seed(0)


@torch.no_grad()
def embed_pil(img):
    face = A.face_from_pil(img)
    if face is None:
        return None
    return A.embed(face).squeeze(0).cpu()


def to_pil(arr):
    return Image.fromarray((arr * 255).astype(np.uint8))


def compute_scores(subset, n):
    data = fetch_lfw_pairs(subset=subset, color=True, resize=1.0,
                           slice_=(slice(0, 250), slice(0, 250)))
    pairs, labels = data.pairs, data.target
    idx = np.random.default_rng(0).permutation(len(pairs))[:n]

    scores, ys, skipped = [], [], 0
    for k, i in enumerate(idx):
        e1 = embed_pil(to_pil(pairs[i][0]))
        e2 = embed_pil(to_pil(pairs[i][1]))
        if e1 is None or e2 is None:
            skipped += 1
            continue
        scores.append(torch.dot(e1, e2).item())
        ys.append(int(labels[i]))
        if (k + 1) % 100 == 0:
            print(f"  [{subset}] {k + 1}/{len(idx)}")
    print(f"  [{subset}] تخطينا {skipped} زوج (ما انكشف وجه)")
    return np.array(scores), np.array(ys)


def best_threshold(scores, ys):
    ths = np.linspace(-0.2, 1.0, 600)
    accs = [((scores >= t).astype(int) == ys).mean() for t in ths]
    return float(ths[int(np.argmax(accs))]), float(max(accs))


def far_frr(scores, ys, th):
    pred = scores >= th
    far = float((pred & (ys == 0)).sum() / max((ys == 0).sum(), 1))    # قبول شخص غلط
    frr = float((~pred & (ys == 1)).sum() / max((ys == 1).sum(), 1))   # رفض الشخص الصح
    return far, frr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    print("1) معايرة threshold على train...")
    s_tr, y_tr = compute_scores("train", args.n)
    th, acc_tr = best_threshold(s_tr, y_tr)
    print(f"   افضل threshold = {th:.3f} (دقة train = {acc_tr:.3%})")

    print("2) الاختبار على test...")
    s_te, y_te = compute_scores("test", args.n)
    acc_te = float(((s_te >= th).astype(int) == y_te).mean())
    far, frr = far_frr(s_te, y_te, th)
    fpr, tpr, _ = roc_curve(y_te, s_te)
    roc_auc = auc(fpr, tpr)

    print("\n===== النتايج (test) =====")
    print(f"Threshold : {th:.3f}")
    print(f"Accuracy  : {acc_te:.3%}")
    print(f"FAR       : {far:.3%}")
    print(f"FRR       : {frr:.3%}")
    print(f"AUC       : {roc_auc:.4f}")

    with open(A.THRESHOLD_FILE, "w") as f:
        json.dump({"threshold": th, "test_accuracy": acc_te, "FAR": far,
                   "FRR": frr, "AUC": float(roc_auc), "n_pairs": args.n}, f, indent=2)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Accept Rate")
    plt.ylabel("True Accept Rate")
    plt.title("ROC - FaceNet (VGGFace2) on LFW")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150)
    print("\nانحفظت: threshold.json و roc_curve.png")


if __name__ == "__main__":
    main()