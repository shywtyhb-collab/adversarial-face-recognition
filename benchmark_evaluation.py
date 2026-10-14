import argparse
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# 1. Device setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
detector = MTCNN(image_size=160, margin=10, device='cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def load_and_align_face(image_path):
    img = Image.open(image_path).convert('RGB')
    face_tensor = detector(img)
    if face_tensor is None:
        raise ValueError(f"No face detected in: {image_path}")
    return face_tensor.to(device)

def fgsm_attack(image_tensor, target_embedding, epsilon):
    if epsilon == 0.0:
        return image_tensor.clone().detach()
    image_adv = image_tensor.clone().detach().unsqueeze(0)
    image_adv.requires_grad = True

    current_embedding = F.normalize(model(image_adv), p=2, dim=1)
    loss = F.cosine_similarity(current_embedding, target_embedding)
    model.zero_grad()
    loss.backward()

    perturbed = image_adv - epsilon * image_adv.grad.data.sign()
    return torch.clamp(perturbed, -1.0, 1.0).squeeze(0).detach()

def pgd_attack(image_tensor, target_embedding, epsilon, alpha=0.01, num_iter=10):
    if epsilon == 0.0:
        return image_tensor.clone().detach()
    original = image_tensor.clone().detach().unsqueeze(0)
    image_adv = original.clone().detach()

    for _ in range(num_iter):
        image_adv.requires_grad = True
        current_embedding = F.normalize(model(image_adv), p=2, dim=1)
        loss = F.cosine_similarity(current_embedding, target_embedding)
        model.zero_grad()
        loss.backward()

        image_adv = image_adv.detach() - alpha * image_adv.grad.data.sign()
        eta = torch.clamp(image_adv - original, min=-epsilon, max=epsilon)
        image_adv = torch.clamp(original + eta, min=-1.0, max=1.0)

    return image_adv.squeeze(0).detach()

def plot_robustness_curve(epsilons, fgsm_scores, pgd_scores, threshold=0.5):
    plt.figure(figsize=(9, 5))
    plt.plot(epsilons, fgsm_scores, marker='o', color='#1f77b4', linewidth=2, label='FGSM (Single-Step)')
    plt.plot(epsilons, pgd_scores, marker='s', color='#d62728', linewidth=2, label='PGD (Iterative 10-Step)')
    plt.axhline(y=threshold, color='green', linestyle='--', linewidth=1.5, label=f'Decision Threshold ({threshold})')

    plt.title('Adversarial Robustness Curve: Cosine Similarity vs. Perturbation (Epsilon)', fontsize=12, fontweight='bold')
    plt.xlabel('Perturbation Strength (Epsilon)', fontsize=11)
    plt.ylabel('Cosine Similarity Score', fontsize=11)
    plt.ylim(-0.8, 1.0)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left', fontsize=10)
    plt.tight_layout()
    plt.savefig('robustness_curve.png', dpi=300)
    plt.close()
    print("[+] Robustness chart saved as: robustness_curve.png")

def main():
    parser = argparse.ArgumentParser(description="Adversarial Robustness Evaluation")
    parser.add_argument("--img1", required=True, help="Reference face image")
    parser.add_argument("--img2", required=True, help="Probe face image")
    args = parser.parse_args()

    face1 = load_and_align_face(args.img1)
    face2 = load_and_align_face(args.img2)

    with torch.no_grad():
        ref_embedding = F.normalize(model(face2.unsqueeze(0)), p=2, dim=1)

    epsilons = [0.0, 0.01, 0.03, 0.05, 0.08, 0.10]
    fgsm_sims = []
    pgd_sims = []
    table_data = []
    threshold = 0.50

    print("\n" + "="*70)
    print(" Running Quantitative Robustness Evaluation Across Perturbation Range")
    print("="*70)

    for eps in epsilons:
        # Evaluate FGSM
        adv_fgsm = fgsm_attack(face1, ref_embedding, eps)
        with torch.no_grad():
            f_sim = F.cosine_similarity(F.normalize(model(adv_fgsm.unsqueeze(0)), p=2, dim=1), ref_embedding).item()
        fgsm_sims.append(f_sim)

        # Evaluate PGD
        adv_pgd = pgd_attack(face1, ref_embedding, eps)
        with torch.no_grad():
            p_sim = F.cosine_similarity(F.normalize(model(adv_pgd.unsqueeze(0)), p=2, dim=1), ref_embedding).item()
        pgd_sims.append(p_sim)

        table_data.append([
            f"{eps:.2f}",
            f"{f_sim:.4f}",
            "ACCEPTED" if f_sim >= threshold else "REJECTED (Evasion)",
            f"{p_sim:.4f}",
            "ACCEPTED" if p_sim >= threshold else "REJECTED (Evasion)"
        ])

    headers = ["Epsilon", "FGSM Similarity", "FGSM Decision", "PGD Similarity", "PGD Decision"]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    plot_robustness_curve(epsilons, fgsm_sims, pgd_sims, threshold=threshold)

if __name__ == "__main__":
    main()
