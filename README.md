# Full package: treinamento e reconhecimento por foto

Pasta `full_package/` contém uma cópia dos scripts necessários para treinar o modelo ResNet50 e executar reconhecimento de imagens por pasta ou câmera.

- `train.py` - Script de treinamento (usa `data/dataset-resized/` por padrão).
- `resnet50.py` - Implementação da arquitetura ResNet50 customizada.
- `pre.py` - Script de predição por lotes; pergunta qual pasta usar (`test/` ou `data/captured/`).
- `ml_model.py` - Integração ML: captura por câmera, salvamento e classificação automática.
- `camera.py` - Captura de imagem via OpenCV e pré-processamento.
- `config.py` - Configurações de câmera e ML.
- `utils.py` - Funções de logging.
- `requirements.txt` - Dependências do projeto.

## Como usar

1. Instale dependências (recomendo criar um virtualenv):

```powershell
pip install -r full_package/requirements.txt
```

2. Para treinar (exemplo):

```powershell
cd full_package
python train.py --train_path ../data/dataset-resized
```

3. Para classificar imagens em lote:

```powershell
cd full_package
python pre.py
```

4. Para testar integração com câmera:

```powershell
cd full_package
python ml_model.py
```

Observação: ajuste o caminho dos pesos no código (`weights-029-0.83.weights.h5`) se necessário.