"""
منحنى المتانة على زوج صور واحد: التشابه مقابل epsilon.

التشغيل:
  python benchmark_evaluation.py --img1 me1.jpg --img2 me2.jpg
ينتج: robustness_curve.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

import attacks as A


def plot_robustness_curve(epsilons, fgsm_scores, pgd_scores, threshold):
    plt.figure(figsize=(9, 5))
    plt.plot(epsilons, fgsm_scores, marker="o", color="#1f77b4", linewidth=2, label="FGSM (Single-Step)")
    plt.plot(epsilons, pgd_scores, marker="s", color="#d62728", linewidth=2, label="PGD (Iterative 10-Step)")
    plt.axhline(y=threshold, color="green", linestyle="--", linewidth=1.5,
                label=f"Decision Threshold ({threshold:.3f})")
    plt.title("Adversarial Robustness Curve: Cosine Similarity vs. Epsilon", fontsize=12, fontweight="bold")
    plt.xlabel("Perturbation Strength (Epsilon, range [-1,1])", fontsize=11)
    plt.ylabel("Cosine Similarity Score", fontsize=11)
    plt.ylim(-0.8, 1.0)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig("robustness_curve.png", dpi=300)
    plt.close()
    print("[+] Robustness chart saved as: robustness_curve.png")


def main():
    parser = argparse.ArgumentParser(description="Adversarial Robustness Evaluation")
    parser.add_argument("--img1", required=True, help="الصورة اللي راح نهاجمها")
    parser.add_argument("--img2", required=True, help="الصورة المرجعية (نفس الشخص)")
    args = parser.parse_args()

    face1 = A.load_face(args.img1)
    face2 = A.load_face(args.img2)
    if face1 is None or face2 is None:
        raise SystemExit("ما انكشف وجه بإحدى الصورتين.")

    th, calibrated = A.load_threshold()
    if not calibrated:
        print("⚠ threshold.json غير موجود. نستخدم 0.5 مؤقتًا. شغّل lfw_eval.py اولًا.")
    ref = A.reference_embedding(face2)

    epsilons = [0.0, 0.01, 0.03, 0.05, 0.08, 0.10]
    fgsm_sims, pgd_sims, table = [], [], []

    print("\n" + "=" * 70)
    print(f" Robustness evaluation | threshold = {th:.3f}")
    print("=" * 70)

    for eps in epsilons:
        f_sim = A.similarity(A.fgsm(face1, ref, eps), ref)
        p_sim = A.similarity(A.pgd(face1, ref, eps), ref)
        fgsm_sims.append(f_sim)
        pgd_sims.append(p_sim)
        table.append([f"{eps:.2f}", f"{A.eps_to_255(eps):.1f}/255",
                      f"{f_sim:.4f}", "ACCEPTED" if f_sim >= th else "REJECTED (Evasion)",
                      f"{p_sim:.4f}", "ACCEPTED" if p_sim >= th else "REJECTED (Evasion)"])

    print(tabulate(table, headers=["Epsilon", "≈ pixels", "FGSM sim", "FGSM decision",
                                   "PGD sim", "PGD decision"], tablefmt="fancy_grid"))
    plot_robustness_curve(epsilons, fgsm_sims, pgd_sims, th)


if __name__ == "__main__":
    main()