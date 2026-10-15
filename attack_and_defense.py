"""
تجربة على زوج صور واحد: هجمات + دفاعات (عادية وتكيّفية).

التشغيل:
  python attack_and_defense.py --img1 me1.jpg --img2 me2.jpg --eps 0.05
"""
import argparse

from tabulate import tabulate

import attacks as A


def main():
    parser = argparse.ArgumentParser(description="Attacks & defenses on FaceNet")
    parser.add_argument("--img1", required=True, help="الصورة اللي راح نهاجمها")
    parser.add_argument("--img2", required=True, help="الصورة المرجعية (نفس الشخص)")
    parser.add_argument("--eps", type=float, default=0.05)
    args = parser.parse_args()

    face1 = A.load_face(args.img1)
    face2 = A.load_face(args.img2)
    if face1 is None or face2 is None:
        raise SystemExit("ما انكشف وجه بإحدى الصورتين. جرّب صورة أوضح.")

    th, calibrated = A.load_threshold()
    ref = A.reference_embedding(face2)
    orig_sim = A.similarity(face1, ref)
    baseline_ok = orig_sim >= th

    print("\n" + "=" * 65)
    print(f"[*] Threshold           : {th:.3f} "
          f"({'معاير على LFW' if calibrated else 'افتراضي! شغّل lfw_eval.py'})")
    print(f"[*] Baseline similarity : {orig_sim:.4f}")
    print(f"[*] Baseline decision   : {'MATCH (Accepted)' if baseline_ok else 'REJECTED'}")
    print(f"[*] Epsilon             : {args.eps} (≈ {A.eps_to_255(args.eps):.1f}/255)")
    print("=" * 65)
    if not baseline_ok:
        print("⚠ النظام يرفض الزوج حتى بدون هجوم، فنتايج الهجوم غير ذات معنى. استخدم زوج آخر.\n")

    eps = args.eps
    pgd_face = A.pgd(face1, ref, eps)
    fgsm_sim = A.similarity(A.fgsm(face1, ref, eps), ref)
    pgd_sim = A.similarity(pgd_face, ref)

    # دفاعات: المهاجم ما يعرف بها (non-adaptive)
    blur_sim = A.similarity(A.blur_defense(pgd_face), ref)
    jpeg_sim = A.similarity(A.jpeg_defense(pgd_face), ref)

    # دفاعات: المهاجم يعرف بها (adaptive)
    ad_blur = A.pgd(face1, ref, eps, preprocess=A.blur_defense)
    ad_blur_sim = A.similarity(A.blur_defense(ad_blur), ref)
    ad_jpeg = A.pgd(face1, ref, eps, preprocess=A.jpeg_bpda)
    ad_jpeg_sim = A.similarity(A.jpeg_defense(ad_jpeg), ref)

    def outcome(s):
        return "ATTACK SUCCESS" if s < th else "ATTACK FAILED"

    rows = [
        ["Clean (no attack)", f"{orig_sim:.4f}", "-", "ACCEPTED" if baseline_ok else "REJECTED"],
        ["FGSM", f"{fgsm_sim:.4f}", f"{orig_sim - fgsm_sim:.4f}", outcome(fgsm_sim)],
        ["PGD (10 steps)", f"{pgd_sim:.4f}", f"{orig_sim - pgd_sim:.4f}", outcome(pgd_sim)],
        ["PGD -> Blur (non-adaptive)", f"{blur_sim:.4f}", f"{orig_sim - blur_sim:.4f}", outcome(blur_sim)],
        ["Adaptive PGD vs Blur", f"{ad_blur_sim:.4f}", f"{orig_sim - ad_blur_sim:.4f}", outcome(ad_blur_sim)],
        ["PGD -> JPEG (non-adaptive)", f"{jpeg_sim:.4f}", f"{orig_sim - jpeg_sim:.4f}", outcome(jpeg_sim)],
        ["Adaptive PGD (BPDA) vs JPEG", f"{ad_jpeg_sim:.4f}", f"{orig_sim - ad_jpeg_sim:.4f}", outcome(ad_jpeg_sim)],
    ]
    print(tabulate(rows, headers=["Condition", "Similarity", "Drop from clean", "Outcome"],
                   tablefmt="fancy_grid"))
    print("\nملاحظة: الدفاع يُعتبر صامد فقط اذا صمد ضد الهجوم التكيّفي، مو العادي.")


if __name__ == "__main__":
    main()