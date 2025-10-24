# Câmera: captura e pré-processamento de imagens usando OpenCV (salva em data/captured)
import cv2
import numpy as np
from typing import Optional
from config import CameraConfig
from utils import log_error, log_success, log_info
import os
import time

def ensure_directory():
    """Garante que o diretório de imagens existe"""
    if not os.path.exists(CameraConfig.IMAGE_SAVE_DIR):
        os.makedirs(CameraConfig.IMAGE_SAVE_DIR)
        log_info(f"Diretório criado: {CameraConfig.IMAGE_SAVE_DIR}")

def list_available_cameras(max_cameras=5):
    """Lista índices de câmeras disponíveis"""
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

def capture_image() -> Optional[np.ndarray]:
    """Captura uma imagem única da câmera"""
    ensure_directory()

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        log_error("Câmera não disponível")
        return None

    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CameraConfig.CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CameraConfig.CAPTURE_HEIGHT)

        # Aquece a câmera
        for _ in range(CameraConfig.CAMERA_WARMUP_ATTEMPTS):
            ret, frame = cap.read()
            if ret:
                break
            time.sleep(CameraConfig.CAMERA_WARMUP_DELAY)
        else:
            log_error("Falha no aquecimento da câmera")
            return None

        # Timer de 3 segundos para estabilização da luz
        log_info("Câmera ativada. Aguardando 3 segundos para estabilização da luz...")
        for i in range(3, 0, -1):
            log_info(f"Capturando em {i} segundo(s)...")
            time.sleep(1)
        
        log_info("📸 Capturando imagem agora!")
        ret, frame = cap.read()
        if not ret or frame is None:
            log_error("Falha ao capturar imagem")
            return None

        processed = preprocess_image(frame)
        if CameraConfig.SAVE_IMAGES and processed is not None:
            timestamp = int(time.time())
            filename_proc = f"{CameraConfig.IMAGE_SAVE_PATH}_{timestamp}_proc.{CameraConfig.IMAGE_FORMAT}"
            cv2.imwrite(filename_proc, (processed * 255).astype("uint8")[:, :, ::-1])  # volta p/ BGR
            log_success(f"Imagem pré-processada salva: {filename_proc}")

        log_info("Imagem capturada com sucesso")
        return processed

    except Exception as e:
        log_error(f"ERRO na captura: {e}")
        return None
    finally:
        cap.release()

def preprocess_image(frame: np.ndarray) -> Optional[np.ndarray]:
    """Pré-processa a imagem para o modelo ML"""
    try:
        img = cv2.resize(frame, (CameraConfig.IMAGE_WIDTH, CameraConfig.IMAGE_HEIGHT))
        '''img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        img = cv2.GaussianBlur(img, CameraConfig.BLUR_KERNEL, 0)'''
        return img
    except Exception as e:
        log_error(f"ERRO no pré-processamento: {e}")
        return None

capture_image()
