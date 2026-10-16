import argparse
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# 1. إعداد المعالج (FaceNet على MPS لتسريع الحسابات، و MTCNN على CPU لتجنب أخطاء التوافق)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"[*] Computation Device: {device}")

detector = MTCNN(image_size=160, margin=10, device='cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def load_and_align_face(image_path):
    img = Image.open(image_path).convert('RGB')
    face_tensor = detector(img)
    if face_tensor is None:
        raise ValueError(f"No face detected in: {image_path}")
    return face_tensor.to(device)

# --- هجوم FGSM (خطوة واحدة) ---
def attack_fgsm(image_tensor, target_embedding, epsilon):
    image_adv = image_tensor.clone().detach().unsqueeze(0)
    image_adv.requires_grad = True

    current_embedding = model(image_adv)
    current_embedding = F.normalize(current_embedding, p=2, dim=1)

    loss = F.cosine_similarity(current_embedding, target_embedding)
    model.zero_grad()
    loss.backward()

    sign_data_grad = image_adv.grad.data.sign()
    perturbed_image = image_adv - epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, -1.0, 1.0)

    return perturbed_image.squeeze(0).detach()

# --- هجوم PGD (خطوات متعددة وإسقاط رياضي) ---
def attack_pgd(image_tensor, target_embedding, epsilon, alpha=0.01, num_iter=10):
    # نبدأ بإضافة تشويش عشوائي خفيف جداً لمنع الوقوع في النقاط الميتة (Local Extrema)
    image_adv = image_tensor.clone().detach().unsqueeze(0)
    noise = torch.FloatTensor(*image_adv.shape).uniform_(-epsilon, epsilon).to(device)
    image_adv = torch.clamp(image_adv + noise, -1.0, 1.0)

    for _ in range(num_iter):
        image_adv.requires_grad = True
        
        current_embedding = model(image_adv)
        current_embedding = F.normalize(current_embedding, p=2, dim=1)

        loss = F.cosine_similarity(current_embedding, target_embedding)
        model.zero_grad()
        loss.backward()

        sign_data_grad = image_adv.grad.data.sign()
        
        # خطوة صغيرة بحجم alpha
        perturbed = image_adv - alpha * sign_data_grad
        
        # Projection (الإسقاط): إجبار التغيير على البقاء ضمن نطاق epsilon من الصورة الأصلية
        eta = torch.clamp(perturbed - image_tensor.unsqueeze(0), min=-epsilon, max=epsilon)
        image_adv = torch.clamp(image_tensor.unsqueeze(0) + eta, min=-1.0, max=1.0).detach()

    return image_adv.squeeze(0)

def save_visual_comparison(orig_tensor, fgsm_tensor, pgd_tensor):
    def tensor_to_np(t):
        img = t.cpu().permute(1, 2, 0).numpy()
        img = (img + 1.0) / 2.0
        return np.clip(img, 0, 1)

    orig_np = tensor_to_np(orig_tensor)
    fgsm_np = tensor_to_np(fgsm_tensor)
    pgd_np = tensor_to_np(pgd_tensor)

    diff_fgsm = np.clip(np.abs(fgsm_np - orig_np) * 10.0, 0, 1)
    diff_pgd = np.clip(np.abs(pgd_np - orig_np) * 10.0, 0, 1)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    axes[0].imshow(orig_np)
    axes[0].set_title("Original (Clean)", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(fgsm_np)
    axes[1].set_title("FGSM Attack", fontsize=11, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(diff_fgsm)
    axes[2].set_title("FGSM Noise (x10)", fontsize=11, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(pgd_np)
    axes[3].set_title("PGD Attack", fontsize=11, fontweight='bold')
    axes[3].axis('off')

    axes[4].imshow(diff_pgd)
    axes[4].set_title("PGD Noise (x10)", fontsize=11, fontweight='bold')
    axes[4].axis('off')

    plt.tight_layout()
    plt.savefig("attacks_comparison_result.png", dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Adversarial Attack Comparison: FGSM vs PGD")
    parser.add_argument("--img1", required=True, help="Path to face image 1")
    parser.add_argument("--img2", required=True, help="Path to face image 2")
    args = parser.parse_args()

    face1 = load_and_align_face(args.img1)
    face2 = load_and_align_face(args.img2)

    with torch.no_grad():
        ref_embedding = model(face2.unsqueeze(0))
        ref_embedding = F.normalize(ref_embedding, p=2, dim=1)

        orig_embedding = model(face1.unsqueeze(0))
        orig_embedding = F.normalize(orig_embedding, p=2, dim=1)
        orig_sim = F.cosine_similarity(orig_embedding, ref_embedding).item()

    threshold = 0.50
    print("\n" + "="*60)
    print(f"[*] Baseline Cosine Similarity : {orig_sim:.4f}")
    print(f"[*] Baseline Decision (Th={threshold}) : {'MATCH (Accepted)' if orig_sim >= threshold else 'REJECT'}")
    print("="*60)

    # المعايير القياسية: 8/255 تعادل تقريباً 0.03
    eps_values = [0.01, 0.03, 0.05, 0.08]
    table_rows = []

    saved_fgsm = None
    saved_pgd = None

    for eps in eps_values:
        # تنفيذ FGSM
        adv_fgsm = attack_fgsm(face1, ref_embedding, eps)
        # تنفيذ PGD (10 خطوات، حجم الخطوة alpha = eps / 4)
        alpha = max(eps / 4.0, 0.005)
        adv_pgd = attack_pgd(face1, ref_embedding, eps, alpha=alpha, num_iter=10)

        with torch.no_grad():
            emb_fgsm = F.normalize(model(adv_fgsm.unsqueeze(0)), p=2, dim=1)
            sim_fgsm = F.cosine_similarity(emb_fgsm, ref_embedding).item()

            emb_pgd = F.normalize(model(adv_pgd.unsqueeze(0)), p=2, dim=1)
            sim_pgd = F.cosine_similarity(emb_pgd, ref_embedding).item()

        status_fgsm = "FAIL" if sim_fgsm >= threshold else "SUCCESS"
        status_pgd = "FAIL" if sim_pgd >= threshold else "SUCCESS"

        table_rows.append([
            f"{eps:.2f}",
            f"{sim_fgsm:.4f} ({status_fgsm})",
            f"{sim_pgd:.4f} ({status_pgd})",
            f"{(sim_fgsm - sim_pgd):.4f}"
        ])

        if eps == 0.05:
            saved_fgsm = adv_fgsm
            saved_pgd = adv_pgd

    headers = ["Epsilon", "FGSM Similarity (Status)", "PGD Similarity (Status)", "PGD Advantage"]
    print("\n" + tabulate(table_rows, headers=headers, tablefmt="fancy_grid"))

    if saved_fgsm is not None and saved_pgd is not None:
        save_visual_comparison(face1, saved_fgsm, saved_pgd)
        print("\n[+] Visual comparison saved as: attacks_comparison_result.png\n")

if __name__ == "__main__":
    main()
