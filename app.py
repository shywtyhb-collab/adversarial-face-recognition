import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import attacks as A

# ضبط إعدادات الصفحة
st.set_page_config(page_title="Biometric Security Evaluator", layout="wide", initial_sidebar_state="expanded")

# مظهر ملكي
st.markdown("""
    <style>
    .main { background-color: #0B1026; color: #F5F1E6; }
    div[data-testid="stSidebar"] { background-color: #121A3E; border-right: 1.5px solid #D4AF37; }
    h1, h2, h3 { color: #D4AF37 !important; font-family: 'Times New Roman', serif; }
    .stButton>button { background-color: #D4AF37 !important; color: #0B1026 !important; font-weight: bold; border-radius: 8px; border: 1.5px solid #D4AF37; }
    .stButton>button:hover { background-color: #890304 !important; color: white !important; border: 1.5px solid #890304; }
    div.stAlert { background-color: #1C2E73; border: 1px solid #D4AF37; color: #F5F1E6; }
    </style>
    """, unsafe_allow_html=True)

# الـ threshold المعاير من lfw_eval.py (او 0.5 اذا ما اجريت المعايرة)
threshold, calibrated = A.load_threshold()


def plot_interactive_curve(face1, ref_emb, current_eps, th):
    eps_range = [0.0, 0.01, 0.03, 0.05, 0.08, 0.10]
    fgsm_list = [A.similarity(A.fgsm(face1, ref_emb, e), ref_emb) for e in eps_range]
    pgd_list = [A.similarity(A.pgd(face1, ref_emb, e), ref_emb) for e in eps_range]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(eps_range, fgsm_list, marker='o', color='#1f77b4', linewidth=2, label='FGSM (Single-Step)')
    ax.plot(eps_range, pgd_list, marker='s', color='#d62728', linewidth=2, label='PGD (Iterative 10-Step)')
    ax.axhline(y=th, color='green', linestyle='--', linewidth=1.5, label=f'Decision Threshold ({th:.3f})')
    ax.axvline(x=current_eps, color='#D4AF37', linestyle=':', linewidth=2, label=f'Current Epsilon ({current_eps})')
    ax.set_title('Adversarial Robustness Curve: Cosine Similarity vs. Perturbation (Epsilon)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Perturbation Strength (Epsilon)', fontsize=11)
    ax.set_ylabel('Cosine Similarity Score', fontsize=11)
    ax.set_ylim(-0.8, 1.0)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left', fontsize=10)
    plt.tight_layout()
    return fig


def tensor_to_display(t):
    img = t.cpu().permute(1, 2, 0).numpy()
    return np.clip((img + 1.0) / 2.0, 0, 1)


# --- العنوان ---
st.title("🛡️ Adversarial Machine Learning Security Suite")
st.subheader("Evaluating Biometric Verification (Face Recognition) Against Adversarial Attacks")

# --- القائمة الجانبية ---
with st.sidebar:
    st.markdown("<div style='text-align: center; font-size: 60px;'>🏛️</div>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Security Console</h3>", unsafe_allow_html=True)
    st.markdown("---")

    file1 = st.file_uploader("Image 1: Original Face", type=["jpeg", "jpg", "png"])
    file2 = st.file_uploader("Image 2: Target Identity", type=["jpeg", "jpg", "png"])

    epsilon = st.slider("Perturbation Strength (Epsilon)", min_value=0.0, max_value=0.15, value=0.05, step=0.01)
    st.caption(f"≈ {A.eps_to_255(epsilon):.1f}/255 بالمدى القياسي للبكسلات")
    attack_type = st.selectbox("Select Attack Algorithm", ["PGD (Iterative Strong)", "FGSM (Single-Step Fast)"])

    run_btn = st.button("🚀 Run Security Benchmark")

    st.markdown("---")
    if calibrated:
        st.caption(f"Threshold (معاير على LFW): {threshold:.3f}")
    else:
        st.warning("threshold.json غير موجود. نستخدم 0.5 مؤقتًا. شغّل lfw_eval.py للمعايرة.")

# --- المساحة الرئيسية ---
if file1 and file2:
    if run_btn:
        with st.spinner("Analyzing Facial Embeddings & Simulating Attacks..."):
            face1 = A.load_face(file1)
            face2 = A.load_face(file2)

            if face1 is None or face2 is None:
                st.error("⚠️ Failed to detect a clear face in one or both of the uploaded images.")
            else:
                ref_emb = A.reference_embedding(face2)
                orig_sim = A.similarity(face1, ref_emb)
                baseline_ok = orig_sim >= threshold          # تصحيح: القرار محسوب مو مكتوب ثابت

                if attack_type == "FGSM (Single-Step Fast)":
                    adv_face = A.fgsm(face1, ref_emb, epsilon)
                else:
                    adv_face = A.pgd(face1, ref_emb, epsilon)
                adv_sim = A.similarity(adv_face, ref_emb)

                st.markdown("---")
                st.markdown("### Visual & Adversarial Results")

                if not baseline_ok:
                    st.warning("النظام يرفض هذا الزوج حتى بدون هجوم، فنتيجة الهجوم غير ذات معنى. جرّب صورتين اوضح لنفس الشخص.")

                orig_np = tensor_to_display(face1)
                adv_np = tensor_to_display(adv_face)
                diff_np = np.clip(np.abs(adv_np - orig_np) * 10.0, 0, 1)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(orig_np, caption=f"Original Face (Baseline Sim: {orig_sim:.4f})", use_container_width=True)
                with col2:
                    st.image(diff_np, caption="Perturbation Map (x10 Amplified)", use_container_width=True)
                with col3:
                    st.image(adv_np, caption=f"Adversarial Image (Attack Sim: {adv_sim:.4f})", use_container_width=True)

                # بطاقات القرار
                base_text = "✔ MATCH (Accepted)" if baseline_ok else "❌ REJECTED (Below Threshold)"
                base_color = "#2ecc71" if baseline_ok else "#e74c3c"
                is_evaded = adv_sim < threshold
                status_text = "❌ REJECTED (Evasion Attack Success)" if is_evaded else "⚠ MATCHED (Attack Failed)"
                status_color = "#e74c3c" if is_evaded else "#2980b9"

                card_col1, card_col2 = st.columns(2)
                with card_col1:
                    st.markdown(f"""
                    <div style="padding:15px; border-radius:10px; background-color:#1B2A6B; border:1px solid #D4AF37;">
                        <h4 style='margin:0; color:#D4AF37;'>Baseline Verification</h4>
                        <p style='margin:5px 0 0 0; font-size:18px;'>Similarity: <b>{orig_sim * 100:.1f}%</b> (threshold {threshold * 100:.1f}%)</p>
                        <p style='margin:5px 0 0 0; color:{base_color}; font-weight:bold;'>{base_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with card_col2:
                    st.markdown(f"""
                    <div style="padding:15px; border-radius:10px; background-color:#1c1024; border:1px solid {status_color};">
                        <h4 style='margin:0; color:{status_color};'>Adversarial Verification</h4>
                        <p style='margin:5px 0 0 0; font-size:18px;'>Similarity: <b>{adv_sim * 100:.1f}%</b> (threshold {threshold * 100:.1f}%)</p>
                        <p style='margin:5px 0 0 0; color:{status_color}; font-weight:bold;'>{status_text}</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("### 📈 Adversarial Robustness Curve Analysis")
                st.pyplot(plot_interactive_curve(face1, ref_emb, epsilon, threshold))
else:
    st.info("👋 Welcome to the Face Recognition Robustness Dashboard. Please upload an image pair in the sidebar and click 'Run Security Benchmark' to generate evaluation results.")