# EDA Utils: funções auxiliares para limpeza, feature engineering e análises
import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from PIL import Image, ImageStat

try:
    import imagehash
except Exception:
    imagehash = None

try:
    from skimage.measure import shannon_entropy
except Exception:
    shannon_entropy = None


CLASSES = ["cardboard","glass","metal","paper","plastic","trash"]

def list_images(dataset_dir: str) -> List[Tuple[str, str]]:
    paths: List[Tuple[str, str]] = []
    base = Path(dataset_dir)
    if not base.exists():
        return paths
    for cls in CLASSES:
        cls_dir = base / cls
        if not cls_dir.exists():
            continue
        for ext in ("*.jpg","*.jpeg","*.png","*.JPG"):
            for p in cls_dir.glob(ext):
                paths.append((cls, str(p)))
    return paths


def is_image_corrupted(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return False
    except Exception:
        return True


def compute_phash(path: str) -> Optional[str]:
    if imagehash is None:
        return None
    try:
        with Image.open(path) as img:
            return str(imagehash.phash(img))
    except Exception:
        return None


def variance_of_laplacian(gray: np.ndarray) -> float:
    if cv2 is None:
        return float("nan")
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def colorfulness_metric(img_bgr: np.ndarray) -> float:
    # Hasler and Süsstrunk colorfulness
    if img_bgr is None:
        return float("nan")
    b, g, r = cv2.split(img_bgr.astype("float32")) if cv2 is not None else (None, None, None)
    if b is None:
        return float("nan")
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))


def compute_features(path: str) -> Dict[str, float]:
    feat: Dict[str, float] = {}
    try:
        with Image.open(path) as pil_img:
            pil_img = pil_img.convert("RGB")
            w, h = pil_img.size
            feat.update({
                "width": w,
                "height": h,
                "aspect_ratio": w / h if h else float("nan"),
                "area_px": w * h,
            })
            stat = ImageStat.Stat(pil_img)
            mean_r, mean_g, mean_b = stat.mean
            std_r, std_g, std_b = stat.stddev
            feat.update({
                "mean_r": mean_r,
                "mean_g": mean_g,
                "mean_b": mean_b,
                "std_r": std_r,
                "std_g": std_g,
                "std_b": std_b,
            })

        if cv2 is not None:
            bgr = cv2.imread(path)
            if bgr is not None:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                sharp = variance_of_laplacian(gray)
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                h_ch, s_ch, v_ch = cv2.split(hsv)
                feat.update({
                    "brightness_mean": float(np.mean(v_ch)),
                    "brightness_std": float(np.std(v_ch)),
                    "saturation_mean": float(np.mean(s_ch)),
                    "saturation_std": float(np.std(s_ch)),
                    "sharpness_lapl_var": sharp,
                    "colorfulness": colorfulness_metric(bgr),
                })

        if shannon_entropy is not None:
            # Entropia de Shannon em escala de cinza
            if cv2 is not None and 'gray' in locals():
                ent = shannon_entropy(gray)
            else:
                with Image.open(path) as pil_img:
                    ent = shannon_entropy(np.array(pil_img.convert('L')))
            feat["entropy_gray"] = float(ent)

    except Exception:
        # Em caso de falha, retorna NaNs para manter o pipeline
        keys = [
            "width","height","aspect_ratio","area_px","mean_r","mean_g","mean_b",
            "std_r","std_g","std_b","brightness_mean","brightness_std","saturation_mean",
            "saturation_std","sharpness_lapl_var","colorfulness","entropy_gray"
        ]
        for k in keys:
            feat.setdefault(k, float("nan"))
    return feat


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
