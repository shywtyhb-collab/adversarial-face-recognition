import argparse
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import cv2
from tabulate import tabulate

# Device setup
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
    image_adv = image_tensor.clone().detach().unsqueeze(0)
    image_adv.requires_grad = True

    current_embedding = model(image_adv)
    current_embedding = F.normalize(current_embedding, p=2, dim=1)

    similarity = F.cosine_similarity(current_embedding, target_embedding)
    loss = similarity

    model.zero_grad()
    loss.backward()

    sign_data_grad = image_adv.grad.data.sign()
    perturbed_image = image_adv - epsilon * sign_data_grad
    return torch.clamp(perturbed_image, -1.0, 1.0).squeeze(0).detach()

def pgd_attack(image_tensor, target_embedding, epsilon, alpha=0.01, num_iter=10):
    """
    Projected Gradient Descent (PGD) Attack
    Multi-step iterative attack that finds stronger adversarial perturbations.
    """
    original_image = image_tensor.clone().detach().unsqueeze(0)
    image_adv = original_image.clone().detach()

    for _ in range(num_iter):
        image_adv.requires_grad = True
        current_embedding = model(image_adv)
        current_embedding = F.normalize(current_embedding, p=2, dim=1)

        loss = F.cosine_similarity(current_embedding, target_embedding)
        model.zero_grad()
        loss.backward()

        sign_data_grad = image_adv.grad.data.sign()
        
        # Take a small step in the gradient direction
        image_adv = image_adv.detach() - alpha * sign_data_grad
        
        # Project back into the epsilon ball around the original image
        eta = torch.clamp(image_adv - original_image, min=-epsilon, max=epsilon)
        image_adv = torch.clamp(original_image + eta, min=-1.0, max=1.0)

    return image_adv.squeeze(0).detach()

def apply_gaussian_blur_defense(image_tensor, kernel_size=3):
    """
    Defense: Preprocessing filter to remove high-frequency adversarial noise.
    """
    img_np = image_tensor.cpu().permute(1, 2, 0).numpy()
    img_blurred = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
    blurred_tensor = torch.from_numpy(img_blurred).permute(2, 0, 1).to(device)
    return blurred_tensor

def main():
    parser = argparse.ArgumentParser(description="Evaluate Attacks & Defenses on FaceNet")
    parser.add_argument("--img1", required=True, help="Path to image 1")
    parser.add_argument("--img2", required=True, help="Path to image 2")
    args = parser.parse_args()

    face1 = load_and_align_face(args.img1)
    face2 = load_and_align_face(args.img2)

    with torch.no_grad():
        ref_embedding = F.normalize(model(face2.unsqueeze(0)), p=2, dim=1)
        orig_embedding = F.normalize(model(face1.unsqueeze(0)), p=2, dim=1)
        orig_sim = F.cosine_similarity(orig_embedding, ref_embedding).item()

    threshold = 0.50
    print("\n" + "="*65)
    print(f"[*] Baseline Cosine Similarity : {orig_sim:.4f}")
    print(f"[*] Baseline Decision (Th={threshold}) : MATCH (Accepted)")
    print("="*65)

    epsilon = 0.05
    print(f"\n[!] Testing Attacks & Defense at Fixed Epsilon = {epsilon} ...\n")

    # 1. FGSM Attack
    fgsm_face = fgsm_attack(face1, ref_embedding, epsilon)
    with torch.no_grad():
        fgsm_sim = F.cosine_similarity(F.normalize(model(fgsm_face.unsqueeze(0)), p=2, dim=1), ref_embedding).item()

    # 2. PGD Attack
    pgd_face = pgd_attack(face1, ref_embedding, epsilon, alpha=0.01, num_iter=10)
    with torch.no_grad():
        pgd_sim = F.cosine_similarity(F.normalize(model(pgd_face.unsqueeze(0)), p=2, dim=1), ref_embedding).item()

    # 3. Defense against PGD (Gaussian Blur)
    defended_face = apply_gaussian_blur_defense(pgd_face, kernel_size=3)
    with torch.no_grad():
        defended_sim = F.cosine_similarity(F.normalize(model(defended_face.unsqueeze(0)), p=2, dim=1), ref_embedding).item()

    results = [
        ["Clean (No Attack)", f"{orig_sim:.4f}", "-", "MATCH (Accepted)"],
        ["FGSM Attack", f"{fgsm_sim:.4f}", f"{orig_sim - fgsm_sim:.4f}", "SUCCESS" if fgsm_sim < threshold else "FAIL"],
        ["PGD Attack (10 steps)", f"{pgd_sim:.4f}", f"{orig_sim - pgd_sim:.4f}", "SUCCESS" if pgd_sim < threshold else "FAIL"],
        ["PGD + Blur Defense", f"{defended_sim:.4f}", f"{orig_sim - defended_sim:.4f}", "DEFENSE WORKED" if defended_sim >= threshold else "DEFENSE BYPASSED"]
    ]

    headers = ["Condition", "Similarity", "Drop from Clean", "Outcome"]
    print(tabulate(results, headers=headers, tablefmt="fancy_grid"))

if __name__ == "__main__":
    main()
