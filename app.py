import streamlit as st
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# 1. إعداد الصفحة
st.set_page_config(page_title="Adversarial Biometrics Evaluator", layout="wide")
st.title("🛡️ Facial Biometrics Security Evaluator")
st.markdown("Evaluating Face Recognition Robustness Against **FGSM** & **PGD** Attacks Across Perturbation Budgets")

# 2. تحميل النماذج والمعالج
@st.cache_resource
def load_models():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    detector = MTCNN(image_size=160, margin=10, device='cpu')
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return device, detector, model

device, detector, model = load_models()

# 3. دوال المعالجة
def process_uploaded_image(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    face = detector(img)
    if face is None:
        return None
    return face.to(device)

def tensor_to_pil(t):
    img = t.cpu().permute(1, 2, 0).numpy()
    img = (img + 1.0) / 2.0
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

def pil_to_tensor(pil_img):
    arr = np.array(pil_img).astype(np.float32) / 255.0
    arr = (arr * 2.0) - 1.0
    t = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).to(device)
    return t

def tensor_to_display(t):
    img = t.cpu().permute(1, 2, 0).numpy()
    img = (img + 1.0) / 2.0
    return np.clip(img, 0, 1)

# 4. خوارزميات الهجوم
def attack_fgsm(image_tensor, target_embedding, epsilon):
    adv = image_tensor.clone().detach().unsqueeze(0)
    adv.requires_grad = True
    emb = F.normalize(model(adv), p=2, dim=1)
    loss = F.cosine_similarity(emb, target_embedding)
    model.zero_grad()
    loss.backward()
    sign_grad = adv.grad.data.sign()
    adv_img = torch.clamp(adv - epsilon * sign_grad, -1.0, 1.0)
    return adv_img.squeeze(0).detach()

def attack_pgd(image_tensor, target_embedding, epsilon, alpha=0.01, num_iter=10):
    adv = image_tensor.clone().detach().unsqueeze(0)
    noise = torch.FloatTensor(*adv.shape).uniform_(-epsilon, epsilon).to(device)
    adv = torch.clamp(adv + noise, -1.0, 1.0)
    for _ in range(num_iter):
        adv.requires_grad = True
        emb = F.normalize(model(adv), p=2, dim=1)
        loss = F.cosine_similarity(emb, target_embedding)
        model.zero_grad()
        loss.backward()
        sign_grad = adv.grad.data.sign()
        perturbed = adv - alpha * sign_grad
        eta = torch.clamp(perturbed - image_tensor.unsqueeze(0), min=-epsilon, max=epsilon)
        adv = torch.clamp(image_tensor.unsqueeze(0) + eta, min=-1.0, max=1.0).detach()
    return adv.squeeze(0)

# 5. دوال الدفاع
def apply_gaussian_blur(tensor_img, radius=1.5):
    pil_img = tensor_to_pil(tensor_img)
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    return pil_to_tensor(blurred)

def apply_jpeg_compression(tensor_img, quality=50):
    pil_img = tensor_to_pil(tensor_img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert("RGB")
    return pil_to_tensor(compressed)

# 6. الشريط الجانبي
st.sidebar.header("⚙️ Evaluation Parameters")
threshold = st.sidebar.slider("Verification Threshold", 0.30, 0.80, 0.50, 0.05)

st.sidebar.subheader("🛡️ Defense Mechanism")
defense_type = st.sidebar.selectbox("Select Defense", ["None", "Gaussian Blur", "JPEG Compression"])
blur_radius = 1.5
jpeg_qual = 50
if defense_type == "Gaussian Blur":
    blur_radius = st.sidebar.slider("Blur Radius", 0.5, 3.0, 1.5, 0.5)
elif defense_type == "JPEG Compression":
    jpeg_qual = st.sidebar.slider("JPEG Quality", 10, 90, 50, 5)

st.sidebar.markdown("---")
uploaded_img1 = st.sidebar.file_uploader("Test Probe Image", type=["jpg", "jpeg", "png"])
uploaded_img2 = st.sidebar.file_uploader("Reference Enrollment Image", type=["jpg", "jpeg", "png"])

if uploaded_img1 and uploaded_img2:
    face1 = process_uploaded_image(uploaded_img1)
    face2 = process_uploaded_image(uploaded_img2)

    if face1 is None or face2 is None:
        st.error("Face not detected clearly. Please upload direct frontal images.")
    else:
        with torch.no_grad():
            ref_emb = F.normalize(model(face2.unsqueeze(0)), p=2, dim=1)
            orig_emb = F.normalize(model(face1.unsqueeze(0)), p=2, dim=1)
            orig_sim = F.cosine_similarity(orig_emb, ref_emb).item()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Clean Cosine Similarity", f"{orig_sim:.4f}")
        with col2:
            st.metric("Baseline Decision", "MATCH (Accepted)" if orig_sim >= threshold else "REJECTED")

        st.markdown("---")
        
        if st.button("🚀 Run Robustness Curve Evaluation"):
            with st.spinner("Generating robustness curve across varying perturbation budgets (Epsilons)..."):
                
                # مجموعة قيم إبسيلون لرسم المنحنى
                eps_list = [0.0, 0.01, 0.03, 0.05, 0.08, 0.12]
                fgsm_sims = []
                pgd_sims = []
                
                sample_fgsm_disp = None
                sample_pgd_disp = None

                for eps in eps_list:
                    if eps == 0.0:
                        fgsm_sims.append(orig_sim)
                        pgd_sims.append(orig_sim)
                        continue

                    # توليد الهجمات
                    adv_f = attack_fgsm(face1, ref_emb, eps)
                    adv_p = attack_pgd(face1, ref_emb, eps, alpha=eps/4.0, num_iter=10)

                    # تطبيق الدفاع
                    if defense_type == "Gaussian Blur":
                        adv_f = apply_gaussian_blur(adv_f, radius=blur_radius)
                        adv_p = apply_gaussian_blur(adv_p, radius=blur_radius)
                    elif defense_type == "JPEG Compression":
                        adv_f = apply_jpeg_compression(adv_f, quality=jpeg_qual)
                        adv_p = apply_jpeg_compression(adv_p, quality=jpeg_qual)

                    # قياس التشابه
                    with torch.no_grad():
                        sim_f = F.cosine_similarity(F.normalize(model(adv_f.unsqueeze(0)), p=2, dim=1), ref_emb).item()
                        sim_p = F.cosine_similarity(F.normalize(model(adv_p.unsqueeze(0)), p=2, dim=1), ref_emb).item()

                    fgsm_sims.append(sim_f)
                    pgd_sims.append(sim_p)

                    # حفظ عينة صور عند eps = 0.05 للمعاينة
                    if eps == 0.05:
                        sample_fgsm_disp = tensor_to_display(adv_f)
                        sample_pgd_disp = tensor_to_display(adv_p)

                # عرض الصور التوضيحية
                st.subheader(f"🖼️ Visual Comparison at ε = 0.05 (Defense: {defense_type})")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(tensor_to_display(face1), caption=f"Original Clean (Sim: {orig_sim:.3f})", use_container_width=True)
                with c2:
                    st.image(sample_fgsm_disp, caption=f"FGSM Attack (Sim: {fgsm_sims[3]:.3f})", use_container_width=True)
                with c3:
                    st.image(sample_pgd_disp, caption=f"PGD Attack (Sim: {pgd_sims[3]:.3f})", use_container_width=True)

                # الرسم البياني الخطي (Robustness Curve)
                st.subheader("📈 Robustness Decay Curve (Line Plot)")
                
                fig, ax = plt.subplots(figsize=(9, 5))
                
                # رسم الخطوط
                ax.plot(eps_list, fgsm_sims, marker='o', linewidth=2.5, color='#2980b9', label=f'FGSM + {defense_type}')
                ax.plot(eps_list, pgd_sims, marker='s', linewidth=2.5, color='#c0392b', label=f'PGD + {defense_type}')
                
                # خط العتبة المتقطع
                ax.axhline(y=threshold, color='#e67e22', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
                
                # تنسيق المحاور
                ax.set_title("System Robustness vs. Adversarial Perturbation (Epsilon)", fontsize=13, fontweight='bold', pad=12)
                ax.set_xlabel("Perturbation Budget (Epsilon ε)", fontsize=11, fontweight='bold')
                ax.set_ylabel("Cosine Similarity", fontsize=11, fontweight='bold')
                ax.set_ylim(-0.8, 1.05)
                ax.set_xticks(eps_list)
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend(loc='upper right', framealpha=0.9)

                # إضافة علامات على نقاط التقاطع
                for x, y in zip(eps_list, fgsm_sims):
                    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=9)
                for x, y in zip(eps_list, pgd_sims):
                    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, -13), ha='center', fontsize=9, color='#c0392b')

                st.pyplot(fig)

                # جدول البيانات الرقمي
                st.subheader("📊 Robustness Decay Data Table")
                df_curve = pd.DataFrame({
                    "Epsilon (ε)": [f"{e:.2f}" for e in eps_list],
                    "FGSM Similarity": [f"{s:.4f}" for s in fgsm_sims],
                    "PGD Similarity": [f"{s:.4f}" for s in pgd_sims],
                    "FGSM Decision": ["Accepted" if s >= threshold else "Rejected (Evasion)" for s in fgsm_sims],
                    "PGD Decision": ["Accepted" if s >= threshold else "Rejected (Evasion)" for s in pgd_sims]
                })
                st.dataframe(df_curve, use_container_width=True)

else:
    st.info("👈 Please upload both face images from the sidebar to start.")
