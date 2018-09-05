# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 14:48:16 2018

@author: murata
"""

import re, random, csv, os
import numpy as np
import keras
from keras.layers import Input, Dense, Flatten, Conv3D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

# original libraries
import preprocess
import readmhd



def make_cnn(input_shape=(257,166,166,1),
             ):
    inputs = Input(shape=input_shape)
    x = Conv3D(filters=8, kernel_size=3, padding='same', activation="relu")(inputs)
    x = Conv3D(filters=8, kernel_size=3, strides=(3,3,3), padding='same', activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=32, kernel_size=3, padding='same', activation="relu")(x)
    x = Conv3D(filters=32, kernel_size=3, strides=(3,3,3), padding='same', activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=64, kernel_size=3, padding='same', activation="relu")(x)
    x = Conv3D(filters=64, kernel_size=3, strides=(3,3,3), padding='same', activation="relu")(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    predictions = Dense(2, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    
    return model


   
def divide_patients(size=(100,100,100),
                    train_per=0.7, val_per=0.15, test_per=0.15):
    path_to_petct = "../../PET-CT_iso3mm/"

    patients=[]
    for patient in os.listdir(path_to_petct):
        xsize, ysize, zsize = preprocess.lung_center_size(patient=patient, center_size="size")
        if xsize<=size[0] and ysize<=size[1] and zsize<=size[2]:
            patients.add(patient)
    train_num = int( train_per*len(patients) )
    validation_num = int( val_per*len(patients) )
#    test_num = len(patients) - train_num - validatin_num
    group_patients={}
    random.shuffle(patients) 
    group_patients["train"] = patients[:train_num]
    group_patients["validation"] = patients[train_num:train_num+validation_num]
    group_patients["test"] = patients[train_num+validation_num:]
    
    return group_patients
    

def make_validation(patients=[],
                    image_size=(100,100,100)):
    path_to_petiso = "../../PET-CT_iso3mm/%s/PETiso.mhd" # % patient
    
#    def load_image_patial(patient,
#                          x_center, y_center, z_center,
#                          ):
        
    
    data, label = np.zeros((len(patients),)+image_size[::-1]), np.zeros((len(patients),2))
    count = 0
    for patient in patients:
        x_center, y_center, z_center = preprocess.lung_center_size(patient=patient, center_size="center")
        xmin, ymin, zmin = max(0, x_center-int(image_size[0]/2.0)), max(0, y_center-int(image_size[1]/2.0)), max(0, z_center-int(image_size[2]/2.0))
        xmax, ymax, zmax = xmin+image_size[0], ymin+image_size[1], zmin+image_size[2]
        pet = readmhd.read(path_to_petiso % patient).vol
        data_temp = pet[zmin:zmax, ymin:ymax, xmin:xmax]
        data[count] = data_temp.reshape(image_size+(1,))
        if re.match("N.*", patient):
            label[count] = np.array([1,0])
        elif re.match("L.*", patient):
            label[count] = np.array([0,1])
        count += 1

    return data, label

def batch_iter(patients=[],
               image_size=(100,100,100),
               steps_per_epoch=2**10,
               batch_size=32,
               ):
    path_to_petiso = "../../PET-CT_iso3mm/%s/PETiso.mhd" # % patient

    while True:
        for step in range(steps_per_epoch):
            data = np.zeros( (batch_size,)+image_size+(1,), dtype=np.uint8 )
            labels = np.zeros( (batch_size,2), dtype=np.uint8 )
            for count in range(batch_size):
                patient = random.choice(patients)
                x_center, y_center, z_center = preprocess.lung_center_size(patient=patient, center_size="center")
                xmin, ymin, zmin = max(0, x_center-int(image_size[0]/2.0)), max(0, y_center-int(image_size[1]/2.0)), max(0, z_center-int(image_size[2]/2.0))
                xmax, ymax, zmax = xmin+image_size[0], ymin+image_size[1], zmin+image_size[2]
                pet = readmhd.read(path_to_petiso % patient).vol
                data_temp = pet[zmin:zmax, ymin:ymax, xmin:xmax]
                data[count] = data_temp.reshape(image_size+(1,))
                if re.match("N.*", patient):
                    labels[count] = np.array([1,0])
                elif re.match("L.*", patient):
                    labels[count] = np.array([0,1])
                count += 1
                if np.random.choice([True,False]): # ランダムに左右反転
                    data = np.flip(data, axis=3)
            yield data, labels
    
        
def train_main(input_shape=(257,166,166,1),
               image_size=(100,100,100),
               steps_per_epoch=2**10,
               epochs=256,
               batch_size=8,
               epochs=256,
               ):
    
    path_to_petiso = "../../PET-CT_iso3mm/%s/PETiso.mhd" # % patient
    path_to_history = "./history.csv"
    
    # divide patients
    matrix_size = list(input_shape[::-1])[1:]
    patients = preprocess.get_patients(matrix_size=matrix_size)
    group_patients = divide_patients(patients, train_per=0.7, val_per=0.15, test_per=0.15)
    
    # load validation data
    data, label = {}, {}
    data["validaiton"], label["validation"] = make_validation(patients=group_patients["validation"], image_size=image_size)
    
    train_gen = batch_iter(patients=group_patients["train"],
                           image_size=image_size, 
                           steps_per_epoch=steps_per_epoch,
                           batch_size=batch_size,
                           )
    
    # load data
#    data, label = {}, {}
#    for group in ["train", "validation", "test"]:
#        data[group], label[group] = make_validation(patients=group_patients[group], size=size)
#        data[group], label[group] = np.zeros((len(group_patients[group]),)+input_shape), np.zeros((len(group_patients[group]),2))
##    data["validation"], label["validation"] = np.zeros((len(patients),)+input_shape+(1,)), np.zeros((len(group_patients["validation"]),)+(1,))
#        count = 0
#        for patient_id in group_patients[group]:
#            volume = readmhd.read(pth_to_petiso % patient_id)
#            data[group][count] = volume.vol.reshape(volume.vol.shape+(1,))
#            if re.match("N.*", patient_id):
#                label[group][count] = np.array([1,0])
#            elif re.match("L.*", patient_id):
#                label[group][count] = np.array([0,1])
#            count += 1
#        data[group] = data[group].reshape(data[group].shape+(1,))
#        label[group] = label[group].reshape(label[group].shape+(1,))
    

    # set cnn model
    input_shape = image_size+(1,)
    model = make_cnn(input_shape=input_shape)
    model.summary()
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  validation_data=(data['validation'], label['validation']),
#                                      validation_steps=valid_steps,
#                                  callbacks=callbacks,
                                  )
    history = model.fit(x=data["train"], y=label["train"], batch_size=batch_size, epochs=epochs, 
              validation_data=(data["validation"], label["validation"]),
              )

    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']

    history_csv = open(path_to_history, 'w')
    writer = csv.writer(history_csv, lineterminator='\n') 
    writer.writerow( ["epoch", "loss", "acc", "val_loss", "val_acc"] ) # headder
    for i in range(len(acc)):
        writer.writerow( [i, loss[i], acc[i], val_loss[i], val_acc[i]] )
    history_csv.close()


def main():
    train_main(epochs=256)
    
    

if __name__ == '__main__':
    main()
