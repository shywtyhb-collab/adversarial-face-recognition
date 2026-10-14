import argparse
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# 1. Device configuration (MPS for FaceNet model, CPU for MTCNN detector)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"[*] Computation Device: {device}")

# 2. Load detector on CPU to avoid MPS pooling errors, and FaceNet model on device
detector = MTCNN(image_size=160, margin=10, device='cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def load_and_align_face(image_path):
    img = Image.open(image_path).convert('RGB')
    face_tensor = detector(img)
    if face_tensor is None:
        raise ValueError(f"No face detected in: {image_path}")
    return face_tensor.to(device)

def fgsm_attack(image_tensor, target_embedding, epsilon):
    image_adv = image_tensor.clone().detach().unsqueeze(0)
    image_adv.requires_grad = True

    current_embedding = model(image_adv)
    current_embedding = F.normalize(current_embedding, p=2, dim=1)

    similarity = F.cosine_similarity(current_embedding, target_embedding)
    loss = similarity

    model.zero_grad()
    loss.backward()

    data_grad = image_adv.grad.data
    sign_data_grad = data_grad.sign()

    perturbed_image = image_adv - epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, -1.0, 1.0)

    return perturbed_image.squeeze(0).detach()

def save_visual_comparison(orig_tensor, adv_tensor):
    def tensor_to_np(t):
        img = t.cpu().permute(1, 2, 0).numpy()
        img = (img + 1.0) / 2.0
        return np.clip(img, 0, 1)

    orig_np = tensor_to_np(orig_tensor)
    adv_np = tensor_to_np(adv_tensor)
    diff_np = np.abs(adv_np - orig_np)
    diff_amplified = np.clip(diff_np * 10.0, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    axes[0].imshow(orig_np)
    axes[0].set_title("Original (Clean Face)", fontsize=13, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(adv_np)
    axes[1].set_title("Adversarial (Perturbed Face)", fontsize=13, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(diff_amplified)
    axes[2].set_title("Perturbation Noise (x10)", fontsize=13, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("adversarial_result.png", dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="FGSM Attack Evaluation on Facial Biometrics")
    parser.add_argument("--img1", required=True, help="Path to image 1")
    parser.add_argument("--img2", required=True, help="Path to image 2")
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
    initial_decision = "MATCH (Accepted)" if orig_sim >= threshold else "NO MATCH (Rejected)"

    print("\n" + "="*50)
    print(f"[*] Baseline Cosine Similarity : {orig_sim:.4f}")
    print(f"[*] Baseline Decision (Th={threshold}) : {initial_decision}")
    print("="*50)

    eps_values = [0.01, 0.03, 0.05, 0.08, 0.12]
    table_rows = []
    best_perturbed_face = None

    for eps in eps_values:
        adv_face = fgsm_attack(face1, ref_embedding, eps)

        with torch.no_grad():
            adv_embedding = model(adv_face.unsqueeze(0))
            adv_embedding = F.normalize(adv_embedding, p=2, dim=1)
            new_sim = F.cosine_similarity(adv_embedding, ref_embedding).item()

        drop = orig_sim - new_sim
        status = "FAIL (Still Matched)" if new_sim >= threshold else "SUCCESS (Evasion Attack)"

        table_rows.append([f"{eps:.2f}", f"{new_sim:.4f}", f"{drop:.4f}", status])

        if best_perturbed_face is None and new_sim < threshold:
            best_perturbed_face = adv_face

    headers = ["Epsilon (Noise)", "Adv Similarity", "Similarity Drop", "Attack Status"]
    print("\n" + tabulate(table_rows, headers=headers, tablefmt="fancy_grid"))

    if best_perturbed_face is not None:
        save_visual_comparison(face1, best_perturbed_face)
        print("\n[+] Visual comparison saved as: adversarial_result.png\n")

if __name__ == "__main__":
    main()
