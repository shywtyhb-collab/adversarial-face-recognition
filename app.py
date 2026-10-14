import streamlit as st
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ضبط إعدادات الصفحة الرسمية بمظهر ملكي داكن
st.set_page_config(page_title="Biometric Security Evaluator", layout="wide", initial_sidebar_state="expanded")

# تطبيق كود CSS مخصص لمنح الواجهة ألواناً ملكية (Royal Dark Blue, Crimson, Gold Accent)
st.markdown("""
    <style>
    .main { background-color: #0B1026; color: #F5F1E6; }
    div[data-testid="stSidebar"] { background-color: #121A3E; border-right: 1.5px solid #D4AF37; }
    h1, h2, h3 { color: #D4AF37 !important; font-family: 'Times New Roman', serif; }
    .stButton>button { background-color: #D4AF37 !important; color: #0B1026 !important; font-weight: bold; border-radius: 8px; border: 1.5px solid #D4AF37; }
    .stButton>button:hover { background-color: #890304 !important; color: white !important; border: 1.5px solid #890304; }
    div.stAlert { background-color: #1C2E73; border: 1px solid #D4AF37; color: #F5F1E6; }
    .stSlider [data-baseweb="slider"] { background-color: #D4AF37; }
    </style>
    """, unsafe_allow_html=True)

# تهيئة النماذج مرة واحدة فقط وحفظها في الذاكرة المؤقتة لسرعة الاستجابة
@st.cache_resource
def load_models():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    detector = MTCNN(image_size=160, margin=10, device='cpu')
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return detector, model, device

detector, model, device = load_models()

def process_uploaded_image(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    face_tensor = detector(img)
    return face_tensor

def fgsm_attack(image_tensor, target_embedding, epsilon):
    image_adv = image_tensor.clone().detach().unsqueeze(0).to(device)
    image_adv.requires_grad = True
    current_embedding = F.normalize(model(image_adv), p=2, dim=1)
    loss = F.cosine_similarity(current_embedding, target_embedding)
    model.zero_grad()
    loss.backward()
    perturbed_image = image_adv - epsilon * image_adv.grad.data.sign()
    return torch.clamp(perturbed_image, -1.0, 1.0).squeeze(0).detach()

def pgd_attack(image_tensor, target_embedding, epsilon, alpha=0.01, num_iter=10):
    original_image = image_tensor.clone().detach().unsqueeze(0).to(device)
    image_adv = original_image.clone().detach()
    for _ in range(num_iter):
        image_adv.requires_grad = True
        current_embedding = F.normalize(model(image_adv), p=2, dim=1)
        loss = F.cosine_similarity(current_embedding, target_embedding)
        model.zero_grad()
        loss.backward()
        image_adv = image_adv.detach() - alpha * image_adv.grad.data.sign()
        eta = torch.clamp(image_adv - original_image, min=-epsilon, max=epsilon)
        image_adv = torch.clamp(original_image + eta, min=-1.0, max=1.0)
    return image_adv.squeeze(0).detach()

# --- عنوان الواجهة التفاعلية ---
st.title("🛡️ Adversarial Machine Learning Security Suite")
st.subheader("Evaluating Biometric Verification (Face Recognition) Against Adversarial Attacks")

# --- القائمة الجانبية لإدخال البيانات والمؤشرات ---
with st.sidebar:
    st.image("https://img.icons8.com/fancy/200/shield.png", width=120)
    st.markdown("### Inputs & Control", unsafe_allow_html=True)
    
    file1 = st.file_uploader("Image 1: Original Face", type=["jpeg", "jpg", "png"])
    file2 = st.file_uploader("Image 2: Target Identity", type=["jpeg", "jpg", "png"])
    
    epsilon = st.slider("Perturbation Strength (Epsilon)", min_value=0.0, max_value=0.15, value=0.05, step=0.01)
    attack_type = st.selectbox("Select Attack Algorithm", ["PGD (Iterative Strong)", "FGSM (Single-Step Fast)"])
    
    run_btn = st.button("🚀 Run Security Benchmark")

# --- المساحة الرئيسية للعرض ---
if file1 and file2:
    if run_btn:
        with st.spinner("Analyzing Facial Embeddings & Simulating Attacks..."):
            face1 = process_uploaded_image(file1)
            face2 = process_uploaded_image(file2)
            
            if face1 is None or face2 is None:
                st.error("⚠️ Failed to detect a clear face in one or both of the uploaded images. Please ensure proper lighting and front-facing poses.")
            else:
                # حساب الميزات الأصلية
                face1 = face1.to(device)
                face2 = face2.to(device)
                with torch.no_grad():
                    ref_emb = F.normalize(model(face2.unsqueeze(0)), p=2, dim=1)
                    orig_emb = F.normalize(model(face1.unsqueeze(0)), p=2, dim=1)
                    orig_sim = F.cosine_similarity(orig_emb, ref_emb).item()
                
                # توليد الهجوم المحدد
                if attack_type == "FGSM (Single-Step Fast)":
                    adv_face = fgsm_attack(face1, ref_emb, epsilon)
                else:
                    adv_face = pgd_attack(face1, ref_emb, epsilon)
                
                # حساب النسبة الجديدة بعد الهجوم
                with torch.no_grad():
                    adv_emb = F.normalize(model(adv_face.unsqueeze(0)), p=2, dim=1)
                    adv_sim = F.cosine_similarity(adv_emb, ref_emb).item()
                
                threshold = 0.50
                
                # --- عرض النتائج التفاعلية ---
                st.markdown("---")
                st.markdown("### Benchmark Results", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                def tensor_to_display(t):
                    img = t.cpu().permute(1, 2, 0).numpy()
                    img = (img + 1.0) / 2.0
                    return np.clip(img, 0, 1)
                
                orig_np = tensor_to_display(face1)
                adv_np = tensor_to_display(adv_face)
                diff_np = np.clip(np.abs(adv_np - orig_np) * 10.0, 0, 1)
                
                with col1:
                    st.image(orig_np, caption=f"Original Face (Baseline Sim: {orig_sim:.4f})", use_container_width=True)
                with col2:
                    st.image(diff_np, caption="Perturbation Noise Map (x10 Amplified)", use_container_width=True)
                with col3:
                    st.image(adv_np, caption=f"Adversarial Image (Attack Sim: {adv_sim:.4f})", use_container_width=True)
                
                # بطاقات الحالة الأمنية
                st.markdown("#### Verification Decision Status", unsafe_allow_html=True)
                card_col1, card_col2 = st.columns(2)
                
                with card_col1:
                    st.markdown(f"""
                    <div style="padding:15px; border-radius:10px; background-color:#1B2A6B; border:1px solid #D4AF37;">
                        <h4 style='margin:0; color:#D4AF37;'>Baseline Verification</h4>
                        <p style='margin:5px 0 0 0; font-size:18px;'>Confidence: <b>{orig_sim*100:.1f}%</b></p>
                        <p style='margin:5px 0 0 0; color:#2ecc71; font-weight:bold;'>✔ MATCH (Accepted)</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with card_col2:
                    is_evaded = adv_sim < threshold
                    status_text = "❌ REJECTED (Evasion Attack Success)" if is_evaded else "⚠ MATCHED (Attack Failed)"
                    status_color = "#e74c3c" if is_evaded else "#2980b9"
                    
                    st.markdown(f"""
                    <div style="padding:15px; border-radius:10px; background-color:#1c1024; border:1px solid {status_color};">
                        <h4 style='margin:0; color:{status_color};'>Adversarial Verification</h4>
                        <p style='margin:5px 0 0 0; font-size:18px;'>Confidence: <b>{adv_sim*100:.1f}%</b></p>
                        <p style='margin:5px 0 0 0; color:{status_color}; font-weight:bold;'>{status_text}</p>
                    </div>
                    """, unsafe_allow_html=True)

else:
    # واجهة افتراضية ترحيبية عند عدم رفع صور
    st.info("👋 Welcome to the Face Recognition Robustness Dashboard. Please upload an image pair in the sidebar and click 'Run Security Benchmark' to generate evaluation results.")
