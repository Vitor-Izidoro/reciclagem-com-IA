# Integração ML: carregamento do modelo, captura via câmera e rotina de classificação automática
import numpy as np
from typing import Optional, Any
import traceback
import os
import time
import cv2
from config import MLConfig
from camera import capture_image
from utils import log_error, log_info
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from resnet50 import ResNet50

# Modelo real usando o ResNet50 treinado
class TrashNetModel:
    def __init__(self, weights_path):
        self.input_shape = (224, 224, 3)  # Modelo foi treinado com 224x224
        self.classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        self.cls_mapping = {
            "cardboard": "PAPELAO",
            "glass": "VIDRO", 
            "metal": "METAL",
            "paper": "PAPEL",
            "plastic": "PLASTICO",
            "trash": "LIXO"
        }
        self.model = self._build_model()
        self.model.load_weights(weights_path)
    
    def _build_model(self):
        inputs = Input(shape=self.input_shape, name='input_1')
        x = ResNet50(input_tensor=inputs)
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(6, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=predictions)
        return model
    
    def predict(self, image_array):
        # Redimensiona para o tamanho esperado pelo modelo (224x224)
        if image_array.shape[:2] != (224, 224):
            image_resized = cv2.resize(image_array, (224, 224))
        else:
            image_resized = image_array
        
        # Expande dimensões para batch
        image_batch = np.expand_dims(image_resized, axis=0)
        
        # Predição
        predictions = self.model.predict(image_batch, verbose=0)
        return predictions[0]

def load_model() -> Optional[Any]:
    """
    Carrega o modelo ML de classificação de resíduos.
    """
    try:        
        weights_path = 'C:\\facul\\6periodo\\lixo-2\\weights\\weights-029-0.83.weights.h5'
        model = TrashNetModel(weights_path)
        log_info("Modelo TrashNet carregado com sucesso")
        return model
        
    except Exception as e:
        log_error(f"Falha ao carregar modelo TrashNet: {e}")
        return None

def save_image_to_test(image_array, filename_prefix="captured"):
    """
    Salva a imagem capturada na pasta test/ para análise
    """
    try:
        # Garante que a pasta test existe
        test_dir = "test"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
            log_info(f"Diretório criado: {test_dir}")
        
        # Gera nome único com timestamp
        timestamp = int(time.time())
        filename = f"{test_dir}/{filename_prefix}_{timestamp}.jpg"
        
        # Converte de RGB para BGR para salvar com OpenCV
        if len(image_array.shape) == 3:
            image_bgr = cv2.cvtColor((image_array * 255).astype('uint8'), cv2.COLOR_RGB2BGR)
        else:
            image_bgr = (image_array * 255).astype('uint8')
        
        # Salva a imagem
        cv2.imwrite(filename, image_bgr)
        log_info(f"Imagem salva em: {filename}")
        return filename
        
    except Exception as e:
        log_error(f"Erro ao salvar imagem na pasta test: {e}")
        return None

def classify_waste() -> Optional[int]:
    """
    Captura uma imagem da câmera, salva na pasta test/, carrega o modelo ML e classifica automaticamente.
    """
    try:
        log_info("=== INICIANDO CLASSIFICAÇÃO AUTOMÁTICA ===")
        
        # 1. Captura a imagem da câmera
        log_info("1. Capturando imagem da câmera...")
        processed_img = capture_image()
        if processed_img is None:
            log_error("Falha na captura da imagem")
            return None

        # 2. Salva a imagem na pasta test/
        log_info("2. Salvando imagem na pasta test/...")
        saved_filename = save_image_to_test(processed_img, "waste_capture")
        if saved_filename is None:
            log_error("Falha ao salvar imagem na pasta test")
            return None

        # 3. Carrega o modelo TrashNet
        log_info("3. Carregando modelo TrashNet...")
        model = load_model()
        if model is None:
            log_error("Falha ao carregar modelo")
            return None

        # 4. Executa predição
        log_info("4. Executando classificação...")
        probabilities = model.predict(processed_img)

        # 5. Encontra a classe com maior probabilidade
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        # Mapeia para o nome da classe do modelo original
        original_class_name = model.classes[predicted_class]
        # Mapeia para o nome usado no sistema atual
        system_class_name = model.cls_mapping.get(original_class_name, original_class_name.upper())

        log_info("=== RESULTADO DA CLASSIFICAÇÃO ===")
        log_info(f"Imagem salva: {saved_filename}")
        log_info(f"Classe detectada: {original_class_name}")
        log_info(f"Sistema: {system_class_name}")
        log_info(f"Confiança: {confidence:.2%}")
        
        # Mostra top 3 predições
        top_indices = np.argsort(probabilities)[::-1][:3]
        log_info("--- Top 3 predições ---")
        for i, idx in enumerate(top_indices):
            class_name = model.classes[idx]
            prob = probabilities[idx]
            log_info(f"{i+1}. {class_name}: {prob:.2%}")

        if confidence >= MLConfig.CONFIDENCE_THRESHOLD:
            log_info(f"✓ Classificação aceita (confiança >= {MLConfig.CONFIDENCE_THRESHOLD:.0%})")
            # Retorna o índice mapeado para o sistema atual
            try:
                system_index = MLConfig.WASTE_TYPES.index(system_class_name)
                return system_index
            except ValueError:
                log_error(f"Classe {system_class_name} não encontrada no sistema")
                return None
        else:
            log_error(f"✗ Baixa confiança ({confidence:.2%}) - abaixo do threshold ({MLConfig.CONFIDENCE_THRESHOLD:.0%})")
            return None

    except Exception as e:
        log_error(f"ERRO na classificação ML: {e}")
        traceback.print_exc()
        return None

# Função para teste do sistema integrado
if __name__ == "__main__":
    log_info("=== TESTE DO SISTEMA INTEGRADO ===")
    log_info("Pressione Enter para capturar e classificar uma imagem...")
    input()
    
    result = classify_waste()
    if result is not None:
        log_info(f"🎯 Resultado final: Classe {result} ({MLConfig.WASTE_TYPES[result]})")
    else:
        log_error("❌ Classificação falhou ou teve baixa confiança")
