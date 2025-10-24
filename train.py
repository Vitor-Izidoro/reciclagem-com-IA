# Treinamento: script para treinar o modelo ResNet50 usando o dataset em data/dataset-resized
from resnet50 import ResNet50 
import os
import tensorflow as tf
import glob

# Escolha interativa de dispositivo
device = input("Digite 'gpu' para treinar na GPU ou 'cpu' para treinar na CPU: ").strip().lower()
if device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print('Treinando na CPU.')
else:
    print('Treinando na GPU (se disponível).')
print('Dispositivos GPU disponíveis:', tf.config.list_physical_devices('GPU'))
import tensorflow as tf

# Escolha: 'gpu' para treinar na GPU, 'cpu' para treinar na CPU
USE_DEVICE = 'gpu'  # altere para 'cpu' se quiser forçar CPU

if USE_DEVICE == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print('Treinando na CPU.')
else:
    # Se quiser usar uma GPU específica, defina o índice, ex: '0'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print('Treinando na GPU (se disponível).')

print('Dispositivos GPU disponíveis:', tf.config.list_physical_devices('GPU'))
import argparse
import numpy as np 
import tensorflow as tf 
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard, ReduceLROnPlateau


os.environ["CUDA_VISIBLE_DEVICES"] = '2'
img_w = 224
img_h = 224
batch_size = 8

def gen(train_path):
    train_datagen = ImageDataGenerator(rotation_range=90,horizontal_flip=True,vertical_flip=True,fill_mode='nearest')
    train_generator = train_datagen.flow_from_directory(directory=train_path,target_size=(img_w,img_h),batch_size=batch_size,class_mode='categorical')


    return train_generator



def bulid_model(input_shape,dropout,fc_layers,num_classes):
    inputs = Input(shape=input_shape,name='input_1')
    x = ResNet50(input_tensor=inputs)
    x = Flatten()(x)
    for fc in fc_layers:
        x = Dense(fc,activation='relu')(x)
        #x = Dropout(dropout)(x)

    predictions = Dense(num_classes,activation='softmax')(x)
    model = Model(inputs=inputs,outputs=predictions)
    model.summary()
    return model


def parse_arguments():

    parser = argparse.ArgumentParser(description='Some parameters.')
    parser.add_argument(
        "--train_path",
        type=str,
        help="Image path.",
        default="trashnet\\data\\dataset-resized"
    )
    return parser.parse_args()


def get_last_checkpoint(weights_dir):
    checkpoints = glob.glob(os.path.join(weights_dir, 'weights-*.weights.h5'))
    if not checkpoints:
        return None, 0
    checkpoints.sort()
    last_ckpt = checkpoints[-1]
    # Extrai o número do epoch do nome do arquivo: weights-005-0.45.weights.h5
    try:
        epoch_num = int(os.path.basename(last_ckpt).split('-')[1])
    except Exception:
        epoch_num = 0
    return last_ckpt, epoch_num

if __name__ == '__main__':

    input_shape = (img_h,img_w,3)
    dropout = 0.2
    fc_layers = [1024,1024]
    num_classes = 6
    epochs = 30
    args = parse_arguments()
    train_path = args.train_path
    train_generator = gen(train_path)
    model = bulid_model(input_shape=input_shape,dropout=dropout,fc_layers=fc_layers,num_classes=num_classes)

    # Tenta carregar o último checkpoint salvo
    weights_dir = 'weights'
    last_ckpt, initial_epoch = get_last_checkpoint(weights_dir)
    if last_ckpt:
        print(f'Carregando pesos do checkpoint: {last_ckpt}')
        model.load_weights(last_ckpt)
    else:
        print('Nenhum checkpoint encontrado, começando do zero.')
        initial_epoch = 0
        try:
            pre_trained_weights = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
            model.load_weights(pre_trained_weights,by_name=True)
        except Exception as e:
            print('load pre-trained weights error {}'.format(e))

    for cls,idx in train_generator.class_indices.items():
        print('Class #{} = {}'.format(idx,cls))

    checkpoint = ModelCheckpoint(filepath='weights/weights-{epoch:03d}-{loss:.2f}.weights.h5', monitor='loss', save_best_only=False, save_weights_only=True)
    checkpoint.set_model(model)

    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    lr_reducer = ReduceLROnPlateau(monitor='loss',factor=np.sqrt(0.1),cooldown=0,patience=2,min_lr=0.5e-6)
    earlystopping = EarlyStopping(monitor='loss',patience=5,verbose=1)
    tensorbord = TensorBoard(log_dir='weights/logs',write_graph=True)

    model.fit(
        train_generator,
        steps_per_epoch=1000,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=[checkpoint, lr_reducer, earlystopping, tensorbord]
    )
