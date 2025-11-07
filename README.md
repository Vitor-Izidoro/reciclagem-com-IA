# reciclagem-com-IA

# Projeto de Classificação de Resíduos (TrashNet + Waste-Classification)

Este projeto utiliza o dataset TrashNet e adapta o código do repositório Waste-Classification para treinar e testar uma rede neural convolucional capaz de classificar imagens de resíduos em seis categorias: **cardboard, glass, metal, paper, plastic, trash**.

## Arquivos Principais do Projeto

### Treinamento
- **`train copy.py`** - Script principal de treinamento (versão atualizada e compatível)
  - Suporte a escolha entre CPU/GPU
  - Compatível com TensorFlow/Keras atuais
  - Data augmentation integrado
  
- **`resnet50.py`** - Implementação da arquitetura ResNet50 personalizada

### Análise e Teste  
- **`pre.py`** - Script de predição/classificação de imagens
  - Analisa imagens da pasta `test/`
  
  - Carrega modelo treinado dos pesos salvos
  - Mostra probabilidades para cada classe

### Dataset
- **`data/dataset-resized/`** - Dataset com imagens organizadas por classe
  - `cardboard/` - Imagens de papelão
  - `glass/` - Imagens de vidro  
  - `metal/` - Imagens de metal
  - `paper/` - Imagens de papel
  - `plastic/` - Imagens de plástico
  - `trash/` - Imagens de lixo comum

### Configuração
- **`requirements.txt`** - Dependências do projeto
- **`weights/`** - Pasta onde são salvos os pesos do modelo treinado

## Origem dos Dados

- **TrashNet**: Dataset público com imagens de resíduos, disponível em [TrashNet Dataset](https://huggingface.co/datasets/garythung/trashnet) e [repositório original](https://github.com/garythung/trashnet).
- As imagens estão organizadas em subpastas por classe, já redimensionadas para facilitar o treinamento.

## Base de Código

- **Waste-Classification**: Código adaptado para Python e Keras/TensorFlow, utilizando a arquitetura ResNet50 com camadas finais customizadas.
- O treinamento inclui técnicas de data augmentation (rotação, flip horizontal/vertical) para melhorar a generalização.

## Instalação

1. Clone o projeto e baixe o dataset TrashNet.
2. Instale as dependências:
	```bash
	pip install -r requirements.txt
	```
3. Certifique-se de que o dataset está em `trashnet/data/dataset-resized`.

## Treinamento

Execute o script principal para treinar o modelo:
```bash
python train copy.py
```
No início, escolha se deseja treinar com CPU ou GPU.

Os pesos do modelo serão salvos na pasta `weights/` após cada época.

# EDA (Análise Exploratória de Dados) para TrashNet

Este módulo executa:
- Limpeza (arquivos corrompidos, hash perceptual para duplicatas)
- Feature Engineering (estatísticas de cor, brilho, nitidez, entropia, etc.)
- Análise Univariada (distribuições e densidades por classe)
- Análise Multivariada (correlação e PCA)
- Seleção de Atributos (Mutual Information, RandomForest, Logistic L1)

## Como atendemos aos requisitos

- Limpeza
	- Verificação de arquivos corrompidos com `PIL.Image.verify()` → campo `is_corrupted` em `image_features_with_meta.csv` e contagens em `summary.json`.
	- Duplicatas via pHash (perceptual hash) e agrupamento por distância de Hamming → campo `phash` e `dup_group`.

- Feature engineering
	- Atributos por imagem: `width`, `height`, `area_px`, `aspect_ratio`, `mean_r/g/b`, `std_r/g/b`, `brightness_mean/std (HSV-V)`, `saturation_mean/std (HSV-S)`, `sharpness_lapl_var` (variância do Laplaciano), `colorfulness`, `entropy_gray` (se disponível).
	- Arquivos: `image_features.csv` (somente features) e `image_features_with_meta.csv` (features + metadados).

- Análise Univariada
	- `plots_univariate/class_distribution.png` (contagem de classes).
	- `plots_univariate/kde_*.png` (densidades por classe das principais features). O código evita warnings/erros quando há baixa variância.

- Análise Multivariada (Correlação)
	- `plots_multivariate/correlation_heatmap.png` (correlação de variáveis numéricas).
	- `plots_multivariate/pca_scatter.png` (PCA 2D padronizado), visualizando separabilidade das classes.

- Seleção de atributos
	- Rankings salvos em CSV: `feature_selection/feature_importance_mutual_info.csv`, `feature_importance_random_forest.csv`, `feature_importance_logreg_l1.csv`.

- Outros
	- Robustez: continua a execução sem bibliotecas opcionais (ex.: `scikit-image`, `ImageHash`) preenchendo os campos ausentes com NaN/None.
	- Guardas para colunas ausentes e pouca variância nos gráficos.
	- Saídas organizadas por pastas, com `summary.json` sintetizando a limpeza.

## Como rodar

1) Instale as dependências da EDA (de preferência em um ambiente virtual):

```powershell
pip install -r trashnet/eda/requirements.txt
```

2) Execute o script (gera saídas em `trashnet/eda/outputs` por padrão):

```powershell
python trashnet/eda/eda.py trashnet/data/dataset-resized trashnet/eda/outputs
```

Arquivos gerados principais:
- `image_features.csv` e `image_features_with_meta.csv`
- `summary.json`
- `plots_univariate/*.png`
- `plots_multivariate/correlation_heatmap.png` e `pca_scatter.png`
- `feature_selection/*.csv`

> Observação: Caso alguma biblioteca opcional falte (ex.: OpenCV ou ImageHash), o script continua e registra NaN/None para os recursos correspondentes.

## Teste/Predict

Para testar o modelo treinado, carregue os pesos salvos e utilize o método `predict` em uma imagem de teste.  
Exemplo de código para predição:
```python
from tensorflow.keras.preprocessing import image
import numpy as np

model.load_weights('weights/weights-xxx.weights.h5')  # Substitua pelo arquivo desejado

img = image.load_img('caminho/para/imagem.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0

pred = model.predict(x)
print('Classe prevista:', np.argmax(pred))
```
-x  
### Como rodar o script de teste

1. **Coloque suas imagens de teste** na pasta `test/` (crie a pasta se não existir):
   - Aceita formatos: .jpg, .jpeg, .png, .JPG
   - Exemplo: fotos de lixo reciclável que você quer classificar

2. **Execute o script de predição:**
   
   **Opção 1 - Se `python` funciona normalmente:**
   ```bash
   python pre.py
   ```
   
   **Opção 2 - Se houver problemas com múltiplas instalações Python:**
   ```bash
   C:\Users\vitor\AppData\Local\Programs\Python\Python312\python.exe pre.py
   ```

3. **Resultado:** O script mostrará para cada imagem as 5 classes mais prováveis com suas probabilidades:
   ```
   test\sua_imagem.jpg
    0.846  cardboard
    0.143  paper
    0.007  glass
    0.003  metal
    0.001  plastic
   ```

**Nota:** Se o comando `python` não funcionar, use o caminho completo do Python onde o TensorFlow está instalado.

## Referências

- [TrashNet Dataset & Código](https://github.com/garythung/trashnet)
- [Waste-Classification Código](https://github.com/garythung/waste-classification)

We also need [@e-lab](http://github.com/e-lab)'s [weight-init module](http://github.com/e-lab/torch-toolbox/blob/master/Weight-init/weight-init.lua), which is already included in this repository.

### CUDA support
Because training takes awhile, you will want to use a GPU to get results in a reasonable amount of time. We used CUDA with a GTX 650 Ti with CUDA. To enable GPU acceleration with CUDA, you'll first need to install CUDA 6.5 or higher. Find CUDA installations [here](http://developer.nvidia.com/cuda-downloads).

Then you need to install following Lua packages for CUDA:
- [torch/cutorch](http://github.com/torch/cutorch)
- [torch/cunn](http://github.com/torch/cunn)

You can install these packages by running the following:

```bash
luarocks install cutorch
luarocks install cunn
```

### Python setup
Python is currently used for some image preprocessing tasks. The Python dependencies are:
- [NumPy](http://numpy.org)
- [SciPy](http://scipy.org)

You can install these packages by running the following:

```bash
# Install using pip
pip install numpy scipy
```

## Usage

### Step 1: Prepare the data
Unzip `data/dataset-resized.zip`.

If adding more data, then the new files must be enumerated properly and put into the appropriate folder in `data/dataset-original` and then preprocessed. Preprocessing the data involves deleting the `data/dataset-resized` folder and then calling `python resize.py` from `trashnet/data`. This will take around half an hour.

### Step 2: Train the model
TODO

### Step 3: Test the model
TODO

### Step 4: View the results
TODO

## Contributing
1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request

## Acknowledgments
- Thanks to the Stanford CS 229 autumn 2016-2017 teaching staff for a great class!
- [@e-lab](http://github.com/e-lab) for their [weight-init Torch module](http://github.com/e-lab/torch-toolbox/blob/master/Weight-init/weight-init.lua)

## TODOs
- finish the Usage portion of the README
- add specific results (and parameters used) that were achieved after the CS 229 project deadline
- add saving of confusion matrix data and creation of graphic to `plot.lua`
- rewrite the data preprocessing to only reprocess new images if the dimensions have not changed
