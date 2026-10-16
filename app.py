import streamlit as st
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import pandas as pd

# إعداد واجهة الصفحة
st.set_page_config(page_title="Adversarial Biometrics Evaluator", layout="wide")
st.title("🛡️ Adversarial Robustness on Facial Biometrics")
st.markdown("منظومة تقييم صمود أنظمة التحقق من الوجه ضد الهجمات العدائية (FGSM vs PGD)")

# تهيئة النماذج والمعالج
@st.cache_resource
def load_models():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    detector = MTCNN(image_size=160, margin=10, device='cpu')
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return device, detector, model

device, detector, model = load_models()

# توابع المعالجة
def process_uploaded_image(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    face = detector(img)
    if face is None:
        return None
    return face.to(device)

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

def tensor_to_display(t):
    img = t.cpu().permute(1, 2, 0).numpy()
    img = (img + 1.0) / 2.0
    return np.clip(img, 0, 1)

# الشريط الجانبي للإعدادات ورفع الصور
st.sidebar.header("إعدادات الاختبار")
threshold = st.sidebar.slider("عتبة القرار (Verification Threshold)", 0.30, 0.80, 0.50, 0.05)
eps_choice = st.sidebar.select_slider("قوة التشويش (Epsilon ε)", options=[0.01, 0.03, 0.05, 0.08, 0.12], value=0.05)

st.sidebar.markdown("---")
uploaded_img1 = st.sidebar.file_uploader("الصورة الأولى (تحت الاختبار)", type=["jpg", "jpeg", "png"])
uploaded_img2 = st.sidebar.file_uploader("الصورة المرجعية (Reference Face)", type=["jpg", "jpeg", "png"])

if uploaded_img1 and uploaded_img2:
    face1 = process_uploaded_image(uploaded_img1)
    face2 = process_uploaded_image(uploaded_img2)

    if face1 is None or face2 is None:
        st.error("تعذر اكتشاف الوجه بوضوح في إحدى الصورتين، يرجى اختيار صورة واضحة ومباشرة.")
    else:
        # حساب التشابه الأساسي
        with torch.no_grad():
            ref_emb = F.normalize(model(face2.unsqueeze(0)), p=2, dim=1)
            orig_emb = F.normalize(model(face1.unsqueeze(0)), p=2, dim=1)
            orig_sim = F.cosine_similarity(orig_emb, ref_emb).item()

        col_base1, col_base2 = st.columns(2)
        with col_base1:
            st.metric("نسبة التشابه الأصلية (Clean Similarity)", f"{orig_sim:.4f}")
        with col_base2:
            status = "قبول (Match)" if orig_sim >= threshold else "رفض (No Match)"
            st.metric("قرار النظام الأساسي", status)

        st.markdown("---")
        
        if st.button("🚀 بدء الهجوم والمقارنة (Run Attacks)"):
            with st.spinner("جاري تطبيق خوارزميات FGSM و PGD وتحليل الملامح..."):
                # تنفيذ الهجمات
                adv_fgsm = attack_fgsm(face1, ref_emb, eps_choice)
                adv_pgd = attack_pgd(face1, ref_emb, eps_choice, alpha=eps_choice/4.0, num_iter=10)

                with torch.no_grad():
                    emb_fgsm = F.normalize(model(adv_fgsm.unsqueeze(0)), p=2, dim=1)
                    sim_fgsm = F.cosine_similarity(emb_fgsm, ref_emb).item()

                    emb_pgd = F.normalize(model(adv_pgd.unsqueeze(0)), p=2, dim=1)
                    sim_pgd = F.cosine_similarity(emb_pgd, ref_emb).item()

                # عرض الصور
                st.subheader("🖼️ المقارنة البصرية وتأثير التشويش")
                c1, c2, c3 = st.columns(3)
                
                orig_disp = tensor_to_display(face1)
                fgsm_disp = tensor_to_display(adv_fgsm)
                pgd_disp = tensor_to_display(adv_pgd)

                with c1:
                    st.image(orig_disp, caption=f"الأصلية (التطابق: {orig_sim:.2f})", use_container_width=True)
                with c2:
                    st.image(fgsm_disp, caption=f"FGSM Attack (التطابق: {sim_fgsm:.2f})", use_container_width=True)
                with c3:
                    st.image(pgd_disp, caption=f"PGD Attack (التطابق: {sim_pgd:.2f})", use_container_width=True)

                # عرض التشويش مكبراً
                st.subheader("🔬 مقدار التشويش المضاف (Perturbation Noise x10)")
                c_noise1, c_noise2 = st.columns(2)
                diff_fgsm = np.clip(np.abs(fgsm_disp - orig_disp) * 10.0, 0, 1)
                diff_pgd = np.clip(np.abs(pgd_disp - orig_disp) * 10.0, 0, 1)
                with c_noise1:
                    st.image(diff_fgsm, caption="تشويش FGSM (مكبّر 10 مرات)", use_container_width=True)
                with c_noise2:
                    st.image(diff_pgd, caption="تشويش PGD (مكبّر 10 مرات)", use_container_width=True)

                # جدول النتائج
                st.subheader("📊 تحليل النتائج والمقارنة")
                data = [
                    {
                        "نوع الهجوم": "الأصل (بدون هجوم)",
                        "نسبة التشابه": f"{orig_sim:.4f}",
                        "الانخفاض عن الأصل": "0.0000",
                        "حالة الهجوم": "نظام طبيعي",
                        "قرار النظام": "قبول (Accepted)" if orig_sim >= threshold else "رفض"
                    },
                    {
                        "نوع الهجوم": "FGSM (One-step)",
                        "نسبة التشابه": f"{sim_fgsm:.4f}",
                        "الانخفاض عن الأصل": f"{orig_sim - sim_fgsm:.4f}",
                        "حالة الهجوم": "نجاح الهجوم (تهرب)" if sim_fgsm < threshold else "فشل (ما زال مقبولاً)",
                        "قرار النظام": "رفض (Evasion)" if sim_fgsm < threshold else "قبول"
                    },
                    {
                        "نوع الهجوم": "PGD (Multi-step Iterative)",
                        "نسبة التشابه": f"{sim_pgd:.4f}",
                        "الانخفاض عن الأصل": f"{orig_sim - sim_pgd:.4f}",
                        "حالة الهجوم": "نجاح الهجوم (تهرب)" if sim_pgd < threshold else "فشل (ما زال مقبولاً)",
                        "قرار النظام": "رفض (Evasion)" if sim_pgd < threshold else "قبول"
                    }
                ]
                st.dataframe(pd.DataFrame(data), use_container_width=True)
else:
    st.info("👈 يرجى رفع الصورتين من الشريط الجانبي لبدء الاختبار التفاعلي.")
