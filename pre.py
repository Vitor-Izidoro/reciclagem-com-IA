# Predição: script para classificar imagens de uma pasta selecionada (test/ ou data/captured/)
from tensorflow.keras.preprocessing import image

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_WEIGHTS

# Vamos importar as funções necessárias diretamente
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from resnet50 import ResNet50
import glob
import numpy as np

def bulid_model(input_shape, dropout, fc_layers, num_classes):
    inputs = Input(shape=input_shape, name='input_1')
    x = ResNet50(input_tensor=inputs)
    x = Flatten()(x)
    for fc in fc_layers:
        x = Dense(fc, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)
    return model 

if __name__ == '__main__':

    img_h = 224
    img_w = 224
    input_shape = (224,224,3)
    fc_layers = [1024,1024]
    num_classes = 6
    
    # Solicitar ao usuário qual pasta usar
    print("Escolha a pasta das imagens para classificação:")
    print("1 - Pasta test/ (imagens locais)")
    print("2 - Pasta data/captured/ (imagens capturadas pela câmera)")
    
    while True:
        choice = input("Digite sua escolha (1 ou 2): ").strip()
        if choice == '1':
            image_path = 'test/'
            print(f"✅ Selecionada: {image_path}")
            break
        elif choice == '2':
            image_path = 'data/captured/'
            print(f"✅ Selecionada: {image_path}")
            break
        else:
            print("❌ Opção inválida! Digite 1 ou 2.")
    
    # Usar caminho relativo centralizado em config.py
    weights_path = str(DEFAULT_WEIGHTS)  # Melhor modelo (menor loss)
    # Aceitar múltiplos formatos de imagem
    images = glob.glob(image_path+'*.jpg') + glob.glob(image_path+'*.jpeg') + glob.glob(image_path+'*.png') + glob.glob(image_path+'*.JPG')
    
    if not images:
        print(f"❌ Nenhuma imagem encontrada na pasta '{image_path}'")
        print("Verifique se existem imagens nos formatos: .jpg, .jpeg, .png, .JPG")
        sys.exit(1)
    
    print(f"📸 Encontradas {len(images)} imagem(ns) para classificar...")
    
    cls_list = ['cardboard','glass','metal','paper','plastic','trash']
    model = bulid_model(input_shape=input_shape,dropout=0,fc_layers=fc_layers,num_classes=num_classes)
    model.load_weights(weights_path)
    for f in images:
        img = image.load_img(f,target_size=(img_h,img_w))
        if img is None:
            continue

        x = image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
        pred = model.predict(x)[0]
        top_inds = pred.argsort()[::-1][:5]
        
        print(f"\n🔍 Analisando: {os.path.basename(f)}")
        print("📊 Resultados da classificação:")
        for i in top_inds:
            confidence = pred[i] * 100
            print(f'   {confidence:6.2f}% - {cls_list[i]}')
        
        # Mostrar a classificação principal
        best_class = cls_list[top_inds[0]]
        best_confidence = pred[top_inds[0]] * 100
        print(f"🏆 Classificação: {best_class.upper()} ({best_confidence:.2f}%)")
        print("-" * 50)
