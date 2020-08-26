# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:06:32 2020

@author: Mai Van Hoa
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import  keras
import numpy as np
import PIL
from PIL import ImageOps
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import os
from tensorflow.keras.layers import *
import random

'''
file_path = './annotations/trimaps/chihuahua_184.png'
# img = PIL.ImageOps.autocontrast(load_img(file_path))  //nhãn của ảnh là các số 0, 1,2,3 tương ứng thuộc về lớp nào
img = load_img(file_path, color_mode='grayscale')
display(img)
'''


img_size = (160, 160)
epoch = 15 
batch_size = 32
num_classes = 4  # Chú ý nhãn của ảnh là 0, 1, 2, 3


class OxfordPets(keras.utils.Sequence):
    '''
    Every Sequence must implement the __getitem__ and the __len__ methods. 
    If you want to modify your dataset between epochs you may implement on_epoch_end. 
    The method __getitem__ should return a complete batch.
    '''
    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        
    def __len__(self):
        return len(self.target_img_paths) // self.batch_size
    
    def __getitem__(self, idx):        # truy cập mảng trực tiếp thông qua instance
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3, ), dtype="float32")  # tạo mảng chứa giá trị 0 có 4 chiều (batch_size, img_size, img_size, 3)
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        
        y = np.zeros((self.batch_size, ) + self.img_size + (1, ), dtype="uint8")    # tạo mảng mask chứa giá trị 0, có 4 chiều (batch_size, img_size, img_size, 1)
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode='grayscale') # cần chuyển sang ảnh xám 1 chiều, bởi mask trong dữ liệu có 3 chiều
            y[j] = np.expand_dims(img, 2) # mở rộng chiều thứ 3 của ảnh là 1 (mask là ảnh 2 chiều) - axis = 2

        return x, y



def unet(input_height, input_weight):
    inputs = Input((input_height, input_weight, 3))
    
    conv1 = Conv2D(16, (3,3), kernel_initializer='he_normal', padding = 'same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('elu')(conv1)

    conv1 = Conv2D(16, (3,3), kernel_initializer='he_normal', padding = 'same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('elu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, (3,3), kernel_initializer='he_normal', padding = 'same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('elu')(conv2)

    conv2 = Conv2D(32, (3,3), kernel_initializer='he_normal', padding = 'same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('elu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, (3,3), kernel_initializer='he_normal', padding = 'same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('elu')(conv3)
    
    conv3 = Conv2D(64,(3,3), kernel_initializer='he_normal', padding = 'same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('elu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128,(3,3), kernel_initializer='he_normal', padding = 'same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('elu')(conv4)

    conv4 = Conv2D(128,(3,3), kernel_initializer='he_normal', padding = 'same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('elu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, (3,3), kernel_initializer='he_normal', padding = 'same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('elu')(conv5)

    conv5 = Conv2D(256,(3,3), kernel_initializer='he_normal', padding = 'same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('elu')(conv5)
    

    up6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    merge6 = concatenate([up6, conv4], axis = 3)
    conv6 = Conv2D(128,(3,3), kernel_initializer='he_normal', padding = 'same')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('elu')(conv6)

    conv6 = Conv2D(128,(3,3), kernel_initializer='he_normal', padding = 'same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('elu')(conv6)

    up7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    merge7 = concatenate([up7, conv3], axis = 3)
    conv7 = Conv2D(64,(3,3), kernel_initializer='he_normal', padding = 'same')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('elu')(conv7)

    conv7 = Conv2D(64, (3,3), kernel_initializer='he_normal', padding = 'same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('elu')(conv7)

    up8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7)
    merge8 = concatenate([up8, conv2], axis = 3)
    conv8 = Conv2D(32,(3,3), kernel_initializer='he_normal', padding = 'same')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('elu')(conv8)

    conv8 = Conv2D(32,(3,3), kernel_initializer='he_normal', padding = 'same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('elu')(conv8)

    up9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8)
    merge9 = concatenate([up9, conv1], axis = 3)
    conv9 = Conv2D(16,(3,3), kernel_initializer='he_normal', padding = 'same')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('elu')(conv9)

    conv9 = Conv2D(16,(3,3), kernel_initializer='he_normal', padding = 'same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('elu')(conv9)
    
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)  # NOTE num_classes, đầu ra là 1 tensor có số chiều bằng số nhãn của ảnh segmentation, mỗi chiều là xác suất thuộc lớp tương ứng

    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
    
    return model

    
def getData(predict=0):
    input_dir = './images/'
    target_dir = './annotations/trimaps'
    
    
    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".jpg")
        ]
    )
    
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )
    
    
    # Split our img paths into a trainning and a validation set
    val_samples = 1000      # number of img validation
    random.Random(1).shuffle(input_img_paths)   # 1 is seed
    random.Random(1).shuffle(target_img_paths)
    
    train_input_img_paths = input_img_paths[: -val_samples]
    train_target_img_paths = target_img_paths[: -val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]
    
    if predict == 1:
        return val_input_img_paths, val_target_img_paths
    
    train_gen = OxfordPets(batch_size, img_size, train_input_img_paths, train_target_img_paths)  # Tạo đối tượng
    val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
    
    return train_gen, val_gen


def predict_Unet():
    '''
    val_input_img_paths, val_target_img_paths = getData(predict=1)
    
    # Generate predictions for all images in the validation set
    model = keras.models.load_model("./Segmentation_animal.h5")
    val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
    val_predicts = model.predict(val_gen)
    
    # Display image, target and mask predict i
    i = 180
    display(Image(filename=val_input_img_paths[i]))
    
    mask_truth = PIL.ImageOps.autocontrast(load_img(val_target_img_paths[i]))
    display(mask_truth)
    
    mask = np.argmax(val_predicts[i], axis=-1)  # lấy giá trị max theo xác suất lớp tương ứng
    mask = np.expand_dims(mask, axis=-1)  # do array_to_img yêu cầu mảng 3 chiều
    mask = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    display(mask)
    '''
    
    # Predict image download from internet
    model = keras.models.load_model("./Segmentation_animal.h5")
    img = load_img('./dogtest.jpg', target_size=img_size)
    display(img)
    img = np.expand_dims(img, axis=0) # yêu cầu mảng 4 chiều, chiều đầu tiên là số ảnh cần predict
    mask_pred = model.predict(img)
    mask_pred = np.argmax(mask_pred[0], axis=-1)  # lấy mask đầu tiên (trng trường hợp cần predict nhiều ảnh)
    mask_pred = np.expand_dims(mask_pred, axis=-1)
    mask_pred = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask_pred))
    display(mask_pred)
    
    
if __name__ == '__main__':
    
    # # Free up RAM in case the model definition cells were run multiple times
    # keras.backend.clear_session()
    # model = unet(160, 160)
    # callbacks = [
    #     keras.callbacks.ModelCheckpoint("Segmentation_animal.h5", save_best_only=True)
    # ]
    
    # train , val = getData()
    # model.fit(train, validation_data= val, epochs=epoch, callbacks=callbacks)  # step_per_epoch = 199 do len(train_gen)= 199, dữ liệu đã lấy một phần cho tập valid
    # '''
    # lấy dữ liệu mỗi batch bằng cách gọi train[idx] ~ __getitem__(idx)
    # '''
    
    predict_Unet()

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    