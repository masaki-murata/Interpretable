# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 14:48:16 2018

@author: murata
"""

import re, random
import numpy as np
import keras
from keras.layers import Input, Dense
from keras.models import Model

# original libraries
import preprocess
import readmhd



def make_cnn(input_shape=(257,166,166,1),
             ):
    inputs = Input(shape=input_shape)
    predictions = Dense(2, activation="softmax")(inputs)
    
    model = Model(inputs=inputs, outputs=predictions)
    
    return model
    
def divide_patients(patients=[],
                    train_per=0.7, val_per=0.15, test_per=0.15):
    train_num = int( train_per*len(patients) )
    validation_num = int( val_per*len(patients) )
#    test_num = len(patients) - train_num - validatin_num
    group_patients={}
    random.shuffle(patients) 
    group_patients["train"] = patients[:train_num]
    group_patients["validation"] = patients[train_num:train_num+validation_num]
    group_patients["test"] = patients[train_num+validation_num:]
    
    return group_patients
    
def train_main(input_shape=(257,166,166),
               ):
    
    pth_to_petiso = "../../PET-CT_iso3mm/%s/PETiso.mhd" # % patient_id
    
    # divide patients
    patients = preprocess.get_patients(voxel_size=list(input_shape[::-1]))
    group_patients = divide_patients(patients, train_per=0.7, val_per=0.15, test_per=0.15)
    
    # load data
    data, label = {}, {}
    for group in ["train", "validation", "test"]:
        data[group], label[group] = np.zeros((len(group_patients[group]),)+input_shape), np.zeros((len(group_patients[group]),))
#    data["validation"], label["validation"] = np.zeros((len(patients),)+input_shape+(1,)), np.zeros((len(group_patients["validation"]),)+(1,))
        count = 0
        for patient_id in group_patients[group]:
            volume = readmhd.read(pth_to_petiso % patient_id)
            data[group][count] = volume.vol
            if re.match("N.*", patient_id):
                label[group][count] = 0
            else:
                label[group][count] = 1
            count += 1
        data[group] = data[group].reshape(data[group].shape+(1,))
        label[group] = label[group].reshape(label[group].shape+(1,))
    
    # set cnn model
    model = make_cnn(input_shape=input_shape)
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x=data["train"], y=label["train"], batch_size=1, epochs=8, validation_data=(data["validation"], label["validation"]))
   
def main():
    train_main()
    
    

if __name__ == '__main__':
    main()
