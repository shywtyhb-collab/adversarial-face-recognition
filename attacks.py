"""
attacks.py - الملف المشترك لكل المشروع (مصدر واحد للحقيقة).

يحتوي:
  1. تحميل النموذج وكاشف الوجه (مرة وحدة)
  2. حساب embedding والتشابه
  3. الهجمات: FGSM و PGD (تدعم هجوم تكيّفي عبر preprocess)
  4. الدفاعات: Gaussian Blur و JPEG
  5. قراءة threshold المعاير من threshold.json

ملاحظة عن المدى: الصور هنا بمدى [-1, 1] (هذا مدى facenet-pytorch).
لذلك epsilon = 0.05 يعادل تقريبًا 6.4/255 بالمدى القياسي للبكسلات.
"""
import io
import json
import os

# يجب قبل استيراد torch: يسمح لعمليات MPS غير المدعومة بالرجوع للـ CPU بدل الانهيار
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# ---------------------------------------------------------------
# 1) النموذج
# ---------------------------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
detector = MTCNN(image_size=160, margin=10, device="cpu")
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
for _p in model.parameters():          # ما نحتاج gradient للاوزان، بس للبكسلات
    _p.requires_grad_(False)

THRESHOLD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "threshold.json")


# ---------------------------------------------------------------
# 2) التحميل والتشابه
# ---------------------------------------------------------------
def face_from_pil(img):
    """صورة PIL -> تنسور الوجه [3,160,160] بمدى [-1,1]، او None اذا ما انكشف وجه"""
    face = detector(img.convert("RGB"))
    return None if face is None else face.to(device)


def load_face(src):
    """src = مسار ملف او ملف مرفوع (Streamlit)"""
    return face_from_pil(Image.open(src))


def embed(x):
    """x: [3,H,W] -> embedding مطبّع [1,512] (يحتفظ بالـ gradient)"""
    return F.normalize(model(x.unsqueeze(0)), p=2, dim=1)


@torch.no_grad()
def reference_embedding(x):
    return embed(x)


@torch.no_grad()
def similarity(x, ref):
    return F.cosine_similarity(embed(x), ref).item()


def eps_to_255(eps):
    """epsilon بمدى [-1,1] -> ما يعادله من 255"""
    return eps * 127.5


# ---------------------------------------------------------------
# 3) threshold
# ---------------------------------------------------------------
def load_threshold(default=0.5):
    """يرجع (threshold, هل_معاير). المعايرة تجي من lfw_eval.py"""
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE) as f:
            return float(json.load(f)["threshold"]), True
    return default, False


# ---------------------------------------------------------------
# 4) الدفاعات
# ---------------------------------------------------------------
def blur_defense(x, kernel_size=3, sigma=0.8):
    """Gaussian Blur قابل للاشتقاق (حتى نقدر نسوي هجوم تكيّفي ضده)"""
    return TF.gaussian_blur(x, [kernel_size, kernel_size], [sigma, sigma])


def jpeg_defense(x, quality=75):
    """ضغط JPEG حقيقي (غير قابل للاشتقاق)"""
    arr = ((x.detach().cpu() + 1) / 2 * 255).clamp(0, 255).round().byte()
    arr = arr.permute(1, 2, 0).numpy()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = np.array(Image.open(buf).convert("RGB"))
    t = torch.from_numpy(out).permute(2, 0, 1).float() / 255 * 2 - 1
    return t.to(x.device)


def jpeg_bpda(x, quality=75):
    """
    BPDA: بالـ forward نمرر عبر JPEG الحقيقي، وبالـ backward نعتبره دالة هوية.
    هذي الطريقة المعروفة لكسر الدفاعات غير القابلة للاشتقاق (Athalye et al. 2018).
    """
    return x + (jpeg_defense(x, quality) - x).detach()


# ---------------------------------------------------------------
# 5) الهجمات (هجوم تهرّب: ننزّل التشابه مع الصورة المرجعية)
# ---------------------------------------------------------------
def _loss(adv, ref, preprocess):
    inp = preprocess(adv) if preprocess is not None else adv
    return F.cosine_similarity(embed(inp), ref).sum()


def fgsm(x, ref, eps, preprocess=None):
    if eps == 0:
        return x.clone().detach()
    adv = x.clone().detach().requires_grad_(True)
    grad, = torch.autograd.grad(_loss(adv, ref, preprocess), adv)
    return (adv.detach() - eps * grad.sign()).clamp(-1, 1)


def pgd(x, ref, eps, steps=10, alpha=None, preprocess=None):
    """
    preprocess=None  -> هجوم عادي (المهاجم ما يعرف بالدفاع)
    preprocess=دفاع  -> هجوم تكيّفي (المهاجم يعرف بالدفاع ويهاجم النظام كامل)
    """
    if eps == 0:
        return x.clone().detach()
    alpha = alpha or 2.5 * eps / steps          # القيمة المعيارية (Madry et al.)
    x = x.detach()
    adv = x.clone()
    for _ in range(steps):
        adv.requires_grad_(True)
        grad, = torch.autograd.grad(_loss(adv, ref, preprocess), adv)
        adv = adv.detach() - alpha * grad.sign()
        adv = torch.min(torch.max(adv, x - eps), x + eps).clamp(-1, 1)   # الاسقاط داخل كرة epsilon
    return adv.detach() 